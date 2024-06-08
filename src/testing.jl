module Testing

using Polynomials4ML: evaluate!, evaluate_ed!, evaluate_ed2!, 
               evaluate, evaluate_d, evaluate_ed, evaluate_dd, evaluate_ed2, 
               _generate_input, AbstractP4MLBasis

using Test, ForwardDiff, Bumper, WithAlloc

using StaticArrays 
using LinearAlgebra: norm 

using ACEbase.Testing: print_tf, println_slim

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


function test_derivatives(basis::AbstractP4MLBasis; 
                          generate_x = () -> _generate_input(basis), 
                          nX = 32, 
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
      x = generate_x()
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
   println() 

   @info("Test consistency of batched evaluation")

   X = [ generate_x() for _ = 1:nX ]
   bP1 = zeros(whatalloc(evaluate!, basis, X)...) 
   bdP1 = deepcopy(bP1)
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
      
   println_slim(@test bP2 ≈ bP1 ≈ bP3 ≈ bP4)
   println_slim(@test bdP3 ≈ bdP1 ≈ bdP4)
   if ed2 
      println_slim(@test bddP4 ≈ bddP1)
   end
   
end

# ------------------------ Test allocations using the WithAlloc interface

function _allocations_inner(basis, x; ed = true, ed2 = true)
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

function test_withalloc(basis::AbstractP4MLBasis; 
            generate_x = () -> _generate_input(basis),
            allowed_allocs = 0, 
            nbatch = 16, 
            kwargs...) 
   X = [ generate_x() for _ = 1:nbatch ]
   x = generate_x()
   match_all = true 
   for Y in (x, X, )
      nalloc = @allocated ( _allocations_inner(basis, Y; kwargs...) )
      print_tf(@test nalloc <= allowed_allocs)
      P1, dP1, ddP1 = evaluate_ed2(basis, Y)
      @no_escape begin 
         P2 = @withalloc evaluate!(basis, Y)
         P3, dP3 = @withalloc evaluate_ed!(basis, Y)
         P4, dP4, ddP4 = @withalloc evaluate_ed2!(basis, Y)
         match_P1P2 = P1 ≈ P2 ≈ P3 ≈ P4
         match_dP1dP2 = dP1 ≈ dP3 ≈ dP4
         match_ddP1ddP2 = ddP1 ≈ ddP4
         print_tf(@test match_P1P2)
         print_tf(@test match_dP1dP2)
         print_tf(@test match_ddP1ddP2)
      end
      if nalloc > allowed_allocs 
         println("nalloc = $nalloc > $allowed_allocs (allowed)")
      end
      if !match_P1P2 || !match_dP1dP2 || !match_ddP1ddP2
         println("standard withalloc evaluations don't match")
      end
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