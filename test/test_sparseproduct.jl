
using BenchmarkTools, Test, Polynomials4ML, ChainRulesCore
using Polynomials4ML: SparseProduct, evaluate, evaluate!, 
         _generate_input, _generate_input_1
using Polynomials4ML.Testing: test_withalloc
using ACEbase.Testing: fdtest, println_slim, print_tf 

test_evaluate(basis::SparseProduct, BB::Tuple{Vararg{AbstractVector}}) = 
      [ prod(BB[j][basis.spec[i][j]] for j = 1:length(BB)) 
      for i = 1:length(basis) ]

test_evaluate(basis::SparseProduct, BB::Tuple{Vararg{AbstractMatrix}}) = 
      [ prod(BB[j][k, basis.spec[i][j]] for j = 1:length(BB)) 
      for k = 1:size(BB[1], 1), i = 1:length(basis)]

function _generate_basis_2(; order=3, len = 50)
   NN = [ rand(10:30) for _ = 1:order ]
   spec = sort([ ntuple(t -> rand(1:NN[t]), order) for _ = 1:len])
   return SparseProduct(spec)
end
             
##

@info("Test serial evaluation")

for ntest = 1:30
   local BB, A1, A2, basis
   order = mod1(ntest, 4)
   basis = _generate_basis_2(; order=order)
   BB = _generate_input_1(basis)
   A1 = test_evaluate(basis, BB)
   A2 = evaluate(basis, BB)
   print_tf(@test A1 ≈ A2)
end

# TODO: revive HyperDualNumbers test later
# A1 = test_evaluate(basis, BB)
# A2 = evaluate(basis, BB)
# hA2 = evaluate(basis, hBB)
# hA2_val = [x.value for x in hA2]

# println_slim(@test A1 ≈ A2 )
# println_slim(@test A2 ≈ hA2_val )

println()

##

@info("Test batch evaluation")

for ntest = 1:30 
   local bBB, bA1, bA2, bA3, basis 

   order = mod1(ntest, 4)
   basis = _generate_basis_2(; order=order)
   bBB = _generate_input(basis)
   bA1 = test_evaluate(basis, bBB)
   bA2 = evaluate(basis, bBB)
   bA3 = copy(bA2)
   evaluate!(bA3, basis, bBB)
   print_tf(@test bA1 ≈ bA2 ≈ bA3)
end

println()

##

@info("    testing withalloc")
basis = _generate_basis_2(; order=2)
BB = _generate_input_1(basis)
bBB = _generate_input(basis)
test_withalloc(basis; batch=false)


##

@info("Testing rrule")
using LinearAlgebra: dot 

for ntest = 1:30
   local bBB, bA2, u, basis, nX 
   order = mod1(ntest, 4)
   basis = _generate_basis_2(; order=order)
   bBB = _generate_input(basis)
   bUU = _generate_input(basis, nX = size(bBB[1], 1))
   _BB(t) = ntuple(i -> bBB[i] + t * bUU[i], order)
   bA2 = evaluate(basis, bBB)
   u = randn(size(bA2))
   F(t) = dot(u, evaluate(basis, _BB(t)))
   dF(t) = begin
      val, pb = rrule(evaluate, basis, _BB(t))
      ∂BB = pb(u)[3]
      return sum( dot(∂BB[i], bUU[i]) for i = 1:length(bUU) )
   end
   print_tf(@test fdtest(F, dF, 0.0; verbose=false))
end
println()

##