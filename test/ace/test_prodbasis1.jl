
using Test, BenchmarkTools, Polynomials4ML, ChainRulesCore
using Polynomials4ML: SimpleProdBasis, release!, SparseSymmProdDAG
using Polynomials4ML.Testing: println_slim, print_tf, generate_SO2_spec
using Random

using ACEbase.Testing: fdtest, dirfdtest
using Lux

P4ML = Polynomials4ML
##

M = 5 
spec = generate_SO2_spec(5, 2*M)
A = randn(ComplexF64, 2*M+1)

## 

@info("Test consistency of SparseSymmetricProduct with SimpleProdBasis")
basis1 = SimpleProdBasis(spec)
AA1 = basis1(A)

basis2 = SparseSymmProdDAG(spec)
AA2 = basis2(A)
proj = basis2.projection 

@info("check against simple implementation")
println_slim(@test AA1 ≈ AA2[proj])

@info("reconstruct spec")
spec_ = P4ML.reconstruct_spec(basis2)
println_slim(@test spec_[proj] == spec)

##

@info("Test with a constant")
spec_c = [ [Int[],]; spec]
basis1_c = SimpleProdBasis(spec_c)
basis2_c = SparseSymmProdDAG(spec_c)
proj_c = basis2_c.projection

spec_c_ = P4ML.reconstruct_spec(basis2_c)
println_slim(@test spec_c_[proj_c] == spec_c)

AA1_c = basis1_c(A)
println_slim(@test AA1 ≈ AA1_c[2:end])
println_slim(@test AA1_c[1] ≈ 1.0)

AA2_c = basis2_c(A)
println_slim(@test AA2_c[1] ≈ 1.0)
println_slim(@test collect(AA2) ≈ collect(AA2_c[2:end]))
println_slim(@test AA2_c[proj_c] ≈ AA1_c)


## 

@info("Test gradient of SparseSymmetricProduct") 

using LinearAlgebra: dot
using Printf

A = randn(2*M+1)
AA = basis2(A)
Δ = randn(length(AA)) ./ (1+length(AA))

f(A) = dot(basis2(A), Δ)
f(A)

δA = randn(length(A)) ./ (1+length(A))
g(t) = f(A + t * δA)

AA, pb = P4ML.rrule(evaluate, basis2, A)
g0 = dot(Δ, AA)
dg0 = dot(pb(Δ)[3], δA)

errs = Float64[]
for h = (0.1).^(0:10)
   push!(errs, abs((g(h) - g0)/h - dg0))
   @printf(" %.2e | %.2e \n", h, errs[end])
end
/(extrema(errs)...)
println_slim(@test /(extrema(errs)...) < 1e-4)


## 

@info("Test consistency of serial and batched evaluation")

nX = 32
bA = randn(ComplexF64, nX, 2*M+1)
bAA1 = zeros(ComplexF64, nX, length(basis2))
for i = 1:nX
   bAA1[i, :] = basis2(bA[i, :])
end
bAA2 = basis2(bA)

println_slim(@test bAA1 ≈ bAA2)

## 

@info("Test consistency of serial and batched evaluation with constant")

nX = 32
bA = randn(ComplexF64, nX, 2*M+1)
bAA1 = zeros(ComplexF64, nX, length(basis2_c))
for i = 1:nX
   bAA1[i, :] = basis2_c(bA[i, :])
end
bAA2 = basis2_c(bA)

println_slim(@test bAA1 ≈ bAA2)

##

sbA = size(bA)
@info("Test batched rrule")
for ntest = 1:30
   local bA, bA2
   local bUU, u
   bA = randn(sbA)
   bU = randn(sbA)
   _BB(t) = bA + t * bU
   bA2 = evaluate(basis2, bA)
   u = randn(size(bA2))
   F(t) = dot(u, evaluate(basis2, _BB(t)))
   dF(t) = begin
      val, pb = rrule(evaluate, basis2, _BB(t))
      ∂BB = pb(u)[3]
      return sum(dot(∂BB[i], bU[i]) for i = 1:length(bU))
   end
   print_tf(@test fdtest(F, dF, 0.0; verbose=false))
end

##

@info("Testing lux interface")

@info("Test consistency of lux and basis")
l_basis2 = P4ML.lux(basis2)
ps, st = Lux.setup(MersenneTwister(1234), l_basis2)
l_AA2, _ = l_basis2(bA, ps, st)
println_slim(@test l_AA2 ≈ basis2(bA))

##

