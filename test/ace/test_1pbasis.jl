
using BenchmarkTools, Test, Polynomials4ML
using Polynomials4ML:  PooledSparseProduct, test_evaluate, evaluate, evaluate!, 
                evalpool, test_evalpool, evalpool!
using ACEbase.Testing: fdtest, println_slim, print_tf

P4ML = Polynomials4ML

N1 = 10 
N2 = 20 
N3 = 50 

B1 = randn(N1)
B2 = randn(N2)
B3 = randn(N3)

spec = sort([ (rand(1:N1), rand(1:N2), rand(1:N3)) for i = 1:100 ])

basis = PooledSparseProduct(spec)

## 

@info("Test serial evaluation")

BB = (B1, B2, B3)

A1 = test_evaluate(basis, BB)
A2 = evaluate(basis, BB)

println_slim(@test A1 ≈ A2 )

## 

@info("Test batched evaluation")
nX = 64 
bBB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )

# using the naive evaluation code 
bA1 = test_evalpool(basis, bBB)
bA2 = evalpool(basis, bBB)

bA3 = copy(bA2)
evalpool!(bA3, basis, bBB)

println_slim(@test bA1 ≈ bA2 ≈ bA3 )


##

@info("Testing _prod_grad")

using StaticArrays, ForwardDiff

prodgrad = P4ML._prod_grad

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

##

@info("Testing _rrule_evalpool")
using LinearAlgebra: dot 

for ntest = 1:30 
   local bBB, bA2 
   bBB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
   bUU = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
   _BB(t) = ( bBB[1] + t * bUU[1], bBB[2] + t * bUU[2], bBB[3] + t * bUU[3] )
   bA2 = evalpool(basis, bBB)
   u = randn(size(bA2))
   F(t) = dot(u, evalpool(basis, _BB(t)))
   dF(t) = begin
      val, pb = P4ML._rrule_evalpool(basis, _BB(t))
      ∂BB = pb(u)
      return sum( dot(∂BB[i], bUU[i]) for i = 1:length(bUU) )
   end
   print_tf(@test fdtest(F, dF, 0.0; verbose=false))
end
println() 