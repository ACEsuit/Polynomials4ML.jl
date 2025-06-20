module Testing

using Polynomials4ML: evaluate!, evaluate_ed!,
               evaluate, evaluate_d, evaluate_ed, 
               pullback, pullback!,
               ka_evaluate!, ka_evaluate_ed!,
               AbstractP4MLBasis

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



# ------------------------ Test correctness of derivatives 

function test_all(basis::AbstractP4MLBasis; evaluate = true, withalloc = true, 
                                            chainrules = true, ka = true) 
   evaluate && test_evaluate_xx(basis)
   withalloc && test_withalloc(basis)
   chainrules && test_chainrules(basis)
   ka && test_ka_evaluate(basis)
   nothing 
end 


function test_derivatives(basis::AbstractP4MLBasis, x::Number)
   P, dP = evaluate_ed(basis, x)
   adP = ForwardDiff.derivative(x -> evaluate(basis, x), x)
   print_tf(@test adP ≈ dP)
end

function test_derivatives(basis::AbstractP4MLBasis, x::AbstractVector)
   P, dP = evaluate_ed(basis, x)
   adP_re = ForwardDiff.jacobian(x -> real.(evaluate(basis, x)), x)
   adP_im = ForwardDiff.jacobian(x -> imag.(evaluate(basis, x)), x)
   adP = adP_re + im * adP_im
   dP2 = [ adP[i, :] for i = 1:size(adP, 1) ]
   print_tf(@test dP2 ≈ dP)
end

"""
This checks a number of things: 
- consistency of evaluate, evaluate_ed
- consistency with evaluate_d and evaluate_dd 
- consistency of single-input and batched evaluation 
- compatibility with ForwardDiff
"""
function test_evaluate_xx(basis::AbstractP4MLBasis; 
                          generate_x = () -> _generate_input(basis), 
                          nX = 15, 
                          ntest = 8, )

   @info("Test consistency of evaluate_**")
   for ntest = 1:ntest 
      x = generate_x()
      P1 = evaluate(basis, x)
      P2, dP2 = evaluate_ed(basis, x)
      dP3 = evaluate_d(basis, x)
      print_tf(@test P1 ≈ P2 )
      print_tf(@test dP2 ≈ dP3)
   end
   println() 

   @info("Test correctness of derivatives")
   for ntest = 1:ntest 
      test_derivatives(basis, generate_x())
   end 
   println() 

   @info("Test consistency of batched evaluation")

   X = [ generate_x() for _ = 1:nX ]
   alc, alcd = whatalloc(evaluate_ed!, basis, X)
   bP1 = zeros(alc...) 
   bdP1 = zeros(alcd...)
   for (i, x) in enumerate(X)
      bP1[i, :] = evaluate(basis, x)
      bdP1[i, :] = evaluate_d(basis, x)
   end
      
   bP2 = evaluate(basis, X)
   bP3, bdP3 = evaluate_ed(basis, X)
      
   println_slim(@test bP2 ≈ bP1 ≈ bP3)
   println_slim(@test bdP3 ≈ bdP1)
   
   return nothing 
end


function test_ka_evaluate(basis::AbstractP4MLBasis; 
                          generate_x = () -> _generate_input(basis), 
                          nXrg = 32:100, 
                          ntest = 8, 
                          dev = Array)
   @info("Testing KA implementation of $basis")
   for _ = 1:ntest                           
      nX = rand(nXrg)                          
      X = [ generate_x() for _ = 1:nX ]
      Xdev = dev(X) 
      P1, dP1 = evaluate_ed(basis, X)
      P2 = dev(similar(P1)) 
      P3 = dev(similar(P1))
      dP3 = dev(similar(dP1))
      ka_evaluate!(P2, basis, X)
      ka_evaluate_ed!(P3, dP3, basis, X)
      print_tf(@test P1 ≈ P2 ≈ P3)
      print_tf(@test dP1 ≈ dP3)   
   end
   println() 
   return nothing 
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
      ∂B = real.(randn(eltype(B), size(B)))
      # we want to differentiate 
      F(t) = sum(∂B .* basis(X + t * dX))
      dF(t) = begin
         val, pb = rrule(evaluate, basis, X + t * dX)
         ∇_X = pb(∂B)[3]
         return sum( sum(a .* b) for (a, b) in zip(∇_X, dX) )
      end
      print_tf(@test fdtest(F, dF, 0.0; verbose = false))
   end    
   println()
   return nothing 
end


# ------------------------ Test allocations using the WithAlloc interface

function _allocations_inner(basis::AbstractP4MLBasis, x; 
                            ed = true)
   @no_escape begin 
      P = @withalloc evaluate!(basis, x)
      s = sum(P)
      if ed 
         P1, dP1 = @withalloc evaluate_ed!(basis, x)
         s += s + sum(P1)
      end 
      nothing 
   end
   return s 
end


function test_withalloc(basis::AbstractP4MLBasis; 
            generate_x = () -> _generate_input(basis),
            generate_batch = () -> _generate_batch(basis),
            allowed_allocs = 0, 
            single = true, 
            ed = true,
            kwargs...) 
   X = generate_batch()
   x = generate_x()
   xX = single ? (x, X) : (X,)
   for Y in xX 
      nalloc1 = _allocations_inner(basis, Y; ed=ed)
      nalloc = @allocated ( _allocations_inner(basis, Y; ed=ed) ) 
      println("nalloc = $nalloc (allowed = $allowed_allocs)")
      print_tf(@test nalloc <= allowed_allocs)
      P1, dP1 = evaluate_ed(basis, Y)
      @no_escape begin 
         P2 = @withalloc evaluate!(basis, Y)
         P3, dP3 = @withalloc evaluate_ed!(basis, Y)
         match_P1P2 = P1 ≈ P2 ≈ P3
         match_dP1dP2 = dP1 ≈ dP3
         match_all = match_P1P2 & match_dP1dP2
         print_tf(@test match_P1P2)
         print_tf(@test match_dP1dP2)
      end
      println() 
   end
   return nothing 
end


end