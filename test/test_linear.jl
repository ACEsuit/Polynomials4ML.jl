using Polynomials4ML
using Random
using LuxCore
using Test
using ACEbase.Testing: println_slim, print_tf, fdtest
using LinearAlgebra
using Optimisers
using ChainRulesCore
using Zygote
# using ChainRulesTestUtils



P4ML = Polynomials4ML
@info("Testing LinearLayer")


in_d, out_d = 4, 3 # feature dimensions
N = 10 # batch size

@info("Testing default constructor is using feature_first = false")
println_slim(@test P4ML.LinearLayer(in_d, out_d; feature_first=false) == P4ML.LinearLayer(in_d, out_d))

println()

##

feature_arr = [true, false]
in_size_arr = [(in_d, N), (N, in_d)]
out_fun_arr = [(x, W) -> W * x, (x, W) -> x * transpose(W)]

##

for (feat, in_size, out_fun) in zip(feature_arr, in_size_arr, out_fun_arr)
   local l, ps, st, x

   @info("Testing feature_first = $feat")
   l = P4ML.LinearLayer(in_d, out_d; feature_first = feat)
   ps, st = LuxCore.setup(MersenneTwister(1234), l)

   if !feat 
      @info("Testing evaluate on vector input vs batch input")
      for ntest = 1:30 
         X = randn(N, in_d) 
         Y1, _ = l(X, ps, st)
         Y2 = hcat([l(X[i,:], ps, st)[1] for i = 1:N]...)'
         Y3 = hcat([ps.W * X[i,:] for i = 1:N]...)'
         print_tf(@test Y1 ≈ Y2 ≈ Y3)
      end
      println() 
   end

   @info("Testing evaluate")
   for ntest = 1:30
      x = randn(in_size)
      out, st = l(x, ps, st)
      print_tf(@test out == out_fun(x, ps.W))
   end
   println()

   @info("Testing rrule")
   for ntest = 1:30
      local x, val, u
      x = randn(in_size)
      bu = randn(in_size)
      _BB(t) = x + t * bu
      val, _ = l(x, ps, st)
      u = randn(size(val))
      F(t) = dot(u, l(_BB(t), ps, st)[1])
      dF(t) = begin
         val, pb = Zygote.pullback(LuxCore.apply, l, _BB(t), ps, st)
         ∂BB = pb((u, st))[2]
         return dot(∂BB, bu)
      end
      print_tf(@test fdtest(F, dF, 0.0; verbose=false))
   end
   println()

   @info("Testing rrule w.r.t ps.W")
   for ntest = 1:30
      local val, W0, re, u 
      w = randn(size(ps.W))
      bu = randn(size(ps.W))
      _BB(t) = w + t * bu
      val, _ = l(x, ps, st)
      W0, re = destructure(ps)

      u = randn(size(val))
      F(t) = dot(u, l(x, re([_BB(t)...]), st)[1])
      dF(t) = begin
         val, pb = Zygote.pullback(LuxCore.apply, l, x, re([_BB(t)...]), st)
         ∂BB = pb((u, st))[3]
         return dot(∂BB[1], bu)
      end
      print_tf(@test fdtest(F, dF, 0.0; verbose=false))
   end

   println()

   # @info("Testing rrule with ChainRulesTestUtils")
   # Why this is failing? Seems some problems with the NamedTuple
   # test_rrule(LuxCore.apply, l, x, ps, st)
end

##

