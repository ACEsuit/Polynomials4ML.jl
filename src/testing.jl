module Testing

using Polynomials4ML: evaluate!, evaluate_ed!, evaluate_ed2!, 
               evaluate, evaluate_d, evaluate_ed, evaluate_dd, evaluate_ed2, 
               pullback!,
               AbstractP4MLBasis, AbstractP4MLTensor, AbstractP4MLLayer

import Polynomials4ML: _generate_input, _generate_batch

using ChainRulesCore: rrule 

using Test, ForwardDiff, Bumper, WithAlloc

using StaticArrays 
using LinearAlgebra: norm, dot 

using ACEbase.Testing: print_tf, println_slim, fdtest 

function time_standard!(P, basis, X)
   for i = 1:length(X)
      Pi = @view P[:, i]
      evaluate!( Pi, basis, X[i] )
   end
   return P 
end

time_batched!(P, basis, X) = evaluate!(P, basis, X)

function time_ed_standard!(P, dP, basis, X)
   for i = 1:length(X)
      evaluate_ed!( (@view P[:, i]), (@view dP[:, i]), basis, X[i] )
   end
   return P, dP  
end

time_ed_batched!(P, dP, basis, X) = evaluate_ed!(P, dP, basis, X)

function time_ed2_standard!(P, dP, ddP, basis, X)
   for i = 1:length(X)
      evaluate_ed2!( (@view P[:, i]), (@view dP[:, i]), (@view ddP[:, i]), basis, X[i] )
   end
   return P, dP  
end

time_ed2_batched!(P, dP, ddP, basis, X) = evaluate_ed2!(P, dP, ddP, basis, X)



# ------------------------ Test correctness of derivatives 

function test_derivatives(basis::AbstractP4MLBasis, x::Number; ed2 = true)
   P, dP = evaluate_ed(basis, x)
   adP = ForwardDiff.derivative(x -> evaluate(basis, x), x)
   print_tf(@test adP ≈ dP)
   
   if ed2 
      P, dP, ddP = evaluate_ed2(basis, x)
      ddP2 = evaluate_dd(basis, x)
      addP = ForwardDiff.derivative(x -> evaluate_d(basis, x), x)
      print_tf(@test addP ≈ ddP ≈ ddP2)
   end
end


function test_derivatives(basis::AbstractP4MLBasis, x::AbstractVector; ed2 = true)
   P, dP = evaluate_ed(basis, x)
   adP_re = ForwardDiff.jacobian(x -> real.(evaluate(basis, x)), x)
   adP_im = ForwardDiff.jacobian(x -> imag.(evaluate(basis, x)), x)
   adP = adP_re + im * adP_im
   dP2 = [ adP[i, :] for i = 1:size(adP, 1) ]
   print_tf(@test dP2 ≈ dP)
   
   if ed2 
      @error("ed2 test for vector inputs not yet implemented")
      # P, dP, ddP = evaluate_ed2(basis, x)
      # ddP2 = evaluate_dd(basis, x)
      # addP = ForwardDiff.gradient(x -> evaluate_d(basis, x), x)
      # print_tf(@test addP ≈ ddP ≈ ddP2)
   end
end

"""
This checks a number of things: 
- consistency of evaluate, evaluate_ed, evaluate_ed2 
- consistency with evaluate_d and evaluate_dd 
- consistency of single-input and batched evaluation 
- compatibility with ForwardDiff
"""
function test_evaluate_xx(basis::AbstractP4MLBasis; 
                          generate_x = () -> _generate_input(basis), 
                          nX = 15, 
                          ntest = 8, 
                          ed2 = true)

   @info("Test consistency of evaluate_**")
   for ntest = 1:ntest 
      x = generate_x()
      P1 = evaluate(basis, x)
      P2, dP2 = evaluate_ed(basis, x)
      dP3 = evaluate_d(basis, x)
      print_tf(@test P1 ≈ P2 )
      print_tf(@test dP2 ≈ dP3)
      if ed2 
         P4, dP4, ddP4 = evaluate_ed2(basis, x)
         ddP5 = evaluate_dd(basis, x)
         print_tf(@test ddP4 ≈ ddP5)
         print_tf(@test P4 ≈ P1)
         print_tf(@test dP4 ≈ dP2)
      end
   end
   println() 

   @info("Test correctness of derivatives")
   for ntest = 1:ntest 
      test_derivatives(basis, generate_x(); ed2 = ed2)
   end 
   println() 

   @info("Test consistency of batched evaluation")

   X = [ generate_x() for _ = 1:nX ]
   alc, alcd = whatalloc(evaluate_ed!, basis, X)
   bP1 = zeros(alc...) 
   bdP1 = zeros(alcd...)
   if ed2 
      bddP1 = deepcopy(bP1)
   end
   for (i, x) in enumerate(X)
      bP1[i, :] = evaluate(basis, x)
      bdP1[i, :] = evaluate_d(basis, x)
      if ed2 
         bddP1[i, :] = evaluate_dd(basis, x)
      end
   end
      
   bP2 = evaluate(basis, X)
   bP3, bdP3 = evaluate_ed(basis, X)
   if ed2 
      bP4, bdP4, bddP4 = evaluate_ed2(basis, X)
   end
      
   println_slim(@test bP2 ≈ bP1 ≈ bP3)
   println_slim(@test bdP3 ≈ bdP1)
   if ed2 
      println_slim(@test bP1 ≈ bP4)
      println_slim(@test bdP1 ≈ bdP4)
      println_slim(@test bddP4 ≈ bddP1)
   end
   
end


function test_chainrules(basis::AbstractP4MLBasis;
                         generate_x = () -> _generate_input(basis), 
                         nX = 15, 
                         ntest = 10)
   @info("Testing rrule(evaluate) - $basis")
   for _ = 1:ntest
      # generate an input batch 
      X = [ generate_x() for _ = 1:nX ]
      B = basis(X)
      # generate a perturbation dX and a tangent ∂B 
      dX = randn(eltype(X), size(X))
      ∂B = randn(eltype(B), size(B))
      # we want to differentiate 
      F(t) = real(sum(∂B .* basis(X + t * dX)))
      dF(t) = begin
         val, pb = rrule(evaluate, basis, X + t * dX)
         ∇_X = pb(conj.(∂B))[3]
         return sum( sum(a .* b) for (a, b) in zip(∇_X, dX) )
      end
      print_tf(@test fdtest(F, dF, 0.0; verbose = false))
   end    
   println()
end


# ------------------------ Test allocations using the WithAlloc interface

function _allocations_inner(basis::AbstractP4MLBasis, x; 
                            ed = true, ed2 = true)
   @no_escape begin 
      P = @withalloc evaluate!(basis, x)
      s = sum(P)
      if ed 
         P1, dP1 = @withalloc evaluate_ed!(basis, x)
         # s += s + sum(P1) + sum(dP1)
      end 
      if ed2 
         P2, dP2, ddP2 = @withalloc evaluate_ed2!(basis, x)
         # s2 = s1 + sum(P2) + sum(dP2) + sum(ddP2)
      end
      nothing 
   end
   return s 
end

_generate_batch(basis::AbstractP4MLBasis; nbatch = rand(7:16)) = 
         [ _generate_input(basis) for _ = 1:nbatch ]

function test_withalloc(basis::AbstractP4MLBasis; 
            generate_x = () -> _generate_input(basis),
            generate_batch = () -> _generate_batch(basis),
            allowed_allocs = 0, 
            ed = true, ed2 = true, 
            kwargs...) 
   X = generate_batch()
   x = generate_x()
   for Y in (x, X, )
      nalloc1 = _allocations_inner(basis, Y; ed=ed, ed2=ed2)
      nalloc = @allocated ( _allocations_inner(basis, Y; ed=ed, ed2=ed2) )
      println("nalloc = $nalloc (allowed = $allowed_allocs)")
      print_tf(@test nalloc <= allowed_allocs)
      P1, dP1 = evaluate_ed(basis, Y)
      if ed2 
         P1, dP1, ddP1 = evaluate_ed2(basis, Y)
      end
      @no_escape begin 
         P2 = @withalloc evaluate!(basis, Y)
         P3, dP3 = @withalloc evaluate_ed!(basis, Y)
         if ed2 
            P4, dP4, ddP4 = @withalloc evaluate_ed2!(basis, Y)
         end
         match_P1P2 = P1 ≈ P2 ≈ P3
         match_dP1dP2 = dP1 ≈ dP3
         match_all = match_P1P2 & match_dP1dP2
         print_tf(@test match_P1P2)
         print_tf(@test match_dP1dP2)
         if ed2
            match_P1P2 = match_P1P2 & (P1 ≈ P4)
            match_dP1dP2 = match_dP1dP2 & (dP3 ≈ dP4)
            match_ddP1ddP2 = ddP1 ≈ ddP4
            math_all = match_P1P2 & match_dP1dP2 & match_ddP1ddP2
            print_tf(@test match_ddP1ddP2)
         end
      end
      println() 
      # if !math_all 
      #    println("standard/withalloc evaluations don't match")
      # end
   end
   return nothing 
end

function _allocations_inner(basis::AbstractP4MLTensor, x; 
                              pb = true)
   @no_escape begin 
      P = @withalloc evaluate!(basis, x)
      s = sum(P)
      if pb 
         T = eltype(P) 
         sz = size(P)
         ∂P = @alloc(T, sz...)
         fill!(∂P, zero(T))
         ∂x = @withalloc pullback!(∂P, basis, x)
      end 
      nothing 
   end
   return s 
end


function test_withalloc(basis::AbstractP4MLTensor; 
            generate_single = () -> _generate_input(basis),
            generate_batch = () -> _generate_batch(basis),
            allowed_allocs = 0, 
            pb = true, 
            batch = true, 
            single = true, 
            kwargs...) 

   if single 
      X = generate_single()
      nalloc_pre = _allocations_inner(basis, X; pb=pb)
      nalloc_pre = _allocations_inner(basis, X; pb=pb)
      nalloc = @allocated ( _allocations_inner(basis, X; pb=pb) )
      println("single: nalloc = $nalloc (allowed = $allowed_allocs)")
      println_slim(@test nalloc <= allowed_allocs)
      A1 = evaluate(basis, X)
      @no_escape begin 
         A2 = @withalloc evaluate!(basis, X)
         match_A1A2 = A1 ≈ A2
         println_slim(@test match_A1A2)
      end
      # if !match_A1A2
      #    println("single: standard withalloc evaluations don't match")
      # end      
   end 

   if batch 
      X = generate_batch()
      nalloc_pre = _allocations_inner(basis, X; pb=pb)
      nalloc = @allocated ( _allocations_inner(basis, X; pb=pb) )
      println("batch: nalloc = $nalloc (allowed = $allowed_allocs)")
      println_slim(@test nalloc <= allowed_allocs)
      A1 = evaluate(basis, X)
      @no_escape begin 
         A2 = @withalloc evaluate!(basis, X)
         match_A1A2 = A1 ≈ A2
         println_slim(@test match_A1A2)
      end
      # if !match_A1A2
      #    println("batch: standard withalloc evaluations don't match")
      # end
   end 

   return nothing 
end



# ------------------------ 
# additional testing utility functions for ACE 

function generate_SO2_spec(order, M, p=1)
   # m = 0, -1, 1, -2, 2, -3, 3, ... 
   i2m(i) = (-1)^(isodd(i-1)) * (i ÷ 2)
   m2i(m) = 2 * abs(m) - (m < 0)

   spec = Vector{Int}[] 

   function append_N!(::Val{N}) where {N} 
      for ci in CartesianIndices(ntuple(_ -> 1:2*M+1, N))
         mm = i2m.(ci.I)
         if (sum(mm) == 0) && (norm(mm, p) <= M) && issorted(ci.I)
            push!(spec, [ci.I...,])
         end
      end
   end


   for N = 1:order 
      append_N!(Val(N))
   end

   return spec 
end 


end