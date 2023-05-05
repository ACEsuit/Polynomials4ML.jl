using Test
using Polynomials4ML.Testing: println_slim, print_tf
using Polynomials4ML: ProductBasis, evaluate, test_evaluate, evaluate_batch!
using LinearAlgebra: norm
using BenchmarkTools
using Polynomials4ML
using ACEbase.Testing: fdtest

N1 = 10
N2 = 20
N3 = 30 

B1 = randn(N1)
B2 = randn(N2)
B3 = randn(N3)

spec = sort([ (rand(1:N1), rand(1:N2), rand(1:N3)) for i = 1:100 ])

basis = ProductBasis(spec)


## 

@info("Test serial evaluation")

BB = (B1, B2, B3)

A1 = test_evaluate(basis, BB)
A2 = evaluate(basis, BB)

println_slim(@test A1 ≈ A2 )

##
@info("Test batch evaluation")

nX = 64 
bBB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
bA1 = zeros(ComplexF64, nX, length(basis))
bA2 = deepcopy(bA1)

for j = 1:nX
    bA1[j, :] = evaluate(basis, (bBB[1][j, :], bBB[2][j, :], bBB[3][j, :]))
end

evaluate_batch!(bA2, basis, bBB)

println_slim(@test bA1 ≈ bA2)


## 
@info("Testing _prod_grad")

using StaticArrays, ForwardDiff

prodgrad = Polynomials4ML._prod_grad

for N = 1:5 
   for ntest = 1:10
      local v1, g 
      b = rand(SVector{3, Float64})
      g = prodgrad(b.data, Val(3))
      g1 = ForwardDiff.gradient(prod, b)
      print_tf(@test g1 ≈ SVector(g...))
   end
end
println() 

@info("Testing _rrule_eval")

# @info("Testing _rrule_eval")
# using LinearAlgebra: dot 


# bBB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
# bUU = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
# _BB(t) = ( bBB[1] + t * bUU[1], bBB[2] + t * bUU[2], bBB[3] + t * bUU[3] )
# bA2 = evaluate(basis, bBB)
# u = randn(size(bA2))
# F(t) = dot(u, evaluate(basis, _BB(t)))
# dF(t) = begin
#     val, pb = Polynomials4ML._rrule_evaluate(basis, _BB(t))
#     ∂BB = pb(u)
#     @show size(∂BB[1])
#     return sum( dot(∂BB[i], bUU[i]) for i = 1:length(bUU) )
# end
# @show dF(0.1)
# print_tf(@test fdtest(F, dF, 0.0; verbose=true))