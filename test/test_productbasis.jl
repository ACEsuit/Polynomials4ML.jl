using Test
using Polynomials4ML.Testing: println_slim, print_tf
using Polynomials4ML: ProductBasis, evaluate, test_evaluate, evaluate_batch!
using LinearAlgebra: norm
using BenchmarkTools
using Polynomials4ML
using ACEbase.Testing: fdtest
using Printf

function grad_test(f, df, X)
   F = f(X) 
   ∇F = df(X)
   nX, nF = size(F)
   U = randn(nX)
   V = randn(nF) ./ (1:nF).^2
   f0 = U' * F * V
   ∇f0 = [ U' * ∇F[i, :, :] * V for i = 1:nX ]
   EE = Matrix(I, (nX, nX))
   for h in 0.1.^(2:10)
      gh = [ (U'*f(X + h * EE[:, i])*V - f0) / h for i = 1:nX ]
      @printf(" %.1e | %.2e \n", h, norm(gh - ∇f0, Inf))
   end
end

N1 = 10
N2 = 5
N3 = 5

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

@info("Testing _rrule_evaluate")
using LinearAlgebra: dot 
bBB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
bUU = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
_BB(t) = ( bBB[1] + t * bUU[1], bBB[2] + t * bUU[2], bBB[3] + t * bUU[3] )
bA2 = Polynomials4ML.evaluate_batch(basis, bBB)
u = randn(size(bA2))
F(t) = dot(u, Polynomials4ML.evaluate_batch(basis, _BB(t)))
dF(t) = begin
    val, pb = Polynomials4ML._rrule_evaluate(basis, _BB(t))
    ∂BB = pb(u)
    return sum( dot(∂BB[i], bUU[i]) for i = 1:length(bUU) )
end
print_tf(@test fdtest(F, dF, 0.0; verbose=true))

println() 