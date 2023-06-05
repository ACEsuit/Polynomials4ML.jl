
using BenchmarkTools, Test, Polynomials4ML
using Polynomials4ML: PooledSparseProduct, evaluate, evaluate!
using ACEbase.Testing: fdtest, println_slim, print_tf

test_evaluate(basis::PooledSparseProduct, BB::Tuple{Vararg{<:AbstractVector}}) =
   [prod(BB[j][basis.spec[i][j]] for j = 1:length(BB))
    for i = 1:length(basis)]

test_evaluate(basis::PooledSparseProduct, BB::Tuple{Vararg{<:AbstractMatrix}}) =
   sum(test_evaluate(basis, ntuple(i -> BB[i][j, :], length(BB)))
       for j = 1:size(BB[1], 1))

P4ML = Polynomials4ML

##

N1 = 10
N2 = 20
N3 = 50

B1 = randn(N1)
B2 = randn(N2)
B3 = randn(N3)

spec = sort([(rand(1:N1), rand(1:N2), rand(1:N3)) for i = 1:100])

basis = PooledSparseProduct(spec)

## 

@info("Test evaluation with a single input (no pooling)")

for _ = 1:30
   local B1
   local B2
   local B3
   local BB
   local A1
   local A2

   B1 = randn(N1)
   B2 = randn(N2)
   B3 = randn(N3)
   BB = (B1, B2, B3)

   A1 = test_evaluate(basis, BB)
   A2 = evaluate(basis, BB)
   print_tf(@test A1 ≈ A2)
end
println()

## 

@info("Test pooling of multiple inputs")
nX = 64

for _ = 1:30
   local bBB
   local bA1
   local bA2
   local bA3
   
   bBB = (randn(nX, N1), randn(nX, N2), randn(nX, N3))

   # using the naive evaluation code 
   bA1 = test_evaluate(basis, bBB)
   bA2 = evaluate(basis, bBB)

   bA3 = copy(bA2)
   evaluate!(bA3, basis, bBB)

   println_slim(@test bA1 ≈ bA2 ≈ bA3)
end


##

@info("Testing _prod_grad")

using StaticArrays, ForwardDiff

prodgrad = P4ML._prod_grad

for N = 1:5
   for ntest = 1:10
      local v1, g
      b = rand(SVector{N,Float64})
      g = prodgrad(b.data, Val(N))
      g1 = ForwardDiff.gradient(prod, b)
      print_tf(@test g1 ≈ SVector(g...))
   end
end
println()

##

@info("Testing _rrule_evalpool")
using LinearAlgebra: dot

for ntest = 1:30
   local bBB, bA2
   local u
   bBB = (randn(nX, N1), randn(nX, N2), randn(nX, N3))
   bUU = (randn(nX, N1), randn(nX, N2), randn(nX, N3))
   _BB(t) = (bBB[1] + t * bUU[1], bBB[2] + t * bUU[2], bBB[3] + t * bUU[3])
   bA2 = evaluate(basis, bBB)
   u = randn(size(bA2))
   F(t) = dot(u, evaluate(basis, _BB(t)))
   dF(t) = begin
      val, pb = P4ML._rrule_evaluate(basis, _BB(t))
      ∂BB = pb(u)
      return sum(dot(∂BB[i], bUU[i]) for i = 1:length(bUU))
   end
   print_tf(@test fdtest(F, dF, 0.0; verbose=false))
end
println()