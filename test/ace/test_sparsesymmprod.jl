
using Test, BenchmarkTools, Polynomials4ML
using Polynomials4ML: SimpleProdBasis, SparseSymmProd
using Polynomials4ML.Testing: println_slim, print_tf, generate_SO2_spec, 
                              test_withalloc
using Random

using ACEbase.Testing: fdtest, dirfdtest
using Lux
using ChainRulesCore: rrule 

P4ML = Polynomials4ML

##

M = 5 
spec = generate_SO2_spec(5, 2*M)
A = randn(ComplexF64, 2*M+1)

## 

@info("Test consistency of SparseSymmetricProduct with SimpleProdBasis")
basis1 = SimpleProdBasis(spec)
AA1 = basis1(A)

basis2 = SparseSymmProd(spec)
AA2 = basis2(A)

@info("check against simple implementation")
println_slim(@test AA1 ≈ AA2)

@info("reconstruct spec")
spec_ = P4ML.reconstruct_spec(basis2)
println_slim(@test spec_ == spec)

println_slim(@test basis2.hasconst == false)

##

@info("Test with a constant")
spec_c = [ [Int[],]; spec]
basis1_c = SimpleProdBasis(spec_c)
basis2_c = SparseSymmProd(spec_c)

println_slim(@test basis2_c.hasconst == true)

spec_c_ = P4ML.reconstruct_spec(basis2_c)
println_slim(@test spec_c_ == spec_c)

AA1_c = basis1_c(A)
println_slim(@test AA1 ≈ AA1_c[2:end])
println_slim(@test AA1_c[1] ≈ 1.0)

AA2_c = basis2_c(A)
println_slim(@test AA2_c[1] ≈ 1.0)
println_slim(@test AA2_c ≈ AA1_c)


## 

@info("Test gradient of SparseSymmetricProduct") 

using LinearAlgebra: dot
using Printf

for ntest = 1:10 
   local A, AA, Δ, f, g, pb, g0, dg0, errs, h, δA

   A = randn(2*M+1)
   AA = basis2(A)
   Δ = randn(length(AA)) ./ (1+length(AA))

   f(A) = dot(basis2(A), Δ)
   f(A)

   δA = randn(length(A)) ./ (1+length(A))
   g(t) = f(A + t * δA)

   AA, pb = rrule(evaluate, basis2, A)
   g0 = dot(Δ, AA)
   dg0 = dot(pb(Δ)[3], δA)

   errs = Float64[]
   for h = (0.1).^(0:10)
      push!(errs, abs((g(h) - g0)/h - dg0))
      # @printf(" %.2e | %.2e \n", h, errs[end])
   end
   /(extrema(errs)...)
   print_tf(@test /(extrema(errs)...) < 1e-4)
end
println() 

## 

@info("Test consistency of serial and batched evaluation")

nX = 32
bA = randn(ComplexF64, nX, 2*M+1)
bAA1 = zeros(ComplexF64, nX, length(spec))
for i = 1:nX
   bAA1[i, :] = basis1(bA[i, :])
end
bAA2 = basis2(bA)

println_slim(@test bAA1 ≈ bAA2)

## 

@info("Test consistency of serial and batched evaluation with constant")

nX = 32
bA = randn(ComplexF64, nX, 2*M+1)
bAA1 = zeros(ComplexF64, nX, length(basis1_c))
for i = 1:nX
   bAA1[i, :] = basis1_c(bA[i, :])
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
println() 

##

#=
@info("Test pb_pb_evaluate")

for ntest = 1:10 
   local A, AA, Δ, Δ², uA, uΔ, F, dF

   A = randn(2*M+1)
   AA = basis2(A)
   Δ = randn(length(AA)) ./ (1:length(AA))
   Δ² = randn(length(A)) ./ (1:length(A))
   uA = randn(length(A)) ./ (1:length(A))
   uΔ = randn(length(AA)) ./ (1:length(AA))

   F(t) = dot(Δ², P4ML.pullback_evaluate(Δ + t * uΔ, basis2, A + t * uA))

   dF(t) = begin 
      val, pb = P4ML.rrule(P4ML.pullback_evaluate, Δ + t * uΔ, basis2,  A + t * uA)
      _, ∇_Δ, _, ∇_A = pb(Δ²)
      return dot(∇_Δ, uΔ) + dot(∇_A, uA)
   end
   print_tf(@test fdtest(F, dF, 0.0; verbose=false))
end
println() 


##

using LinearAlgebra: Diagonal
sbA = size(bA)
@info("Test batched double-pullback")
for ntest = 1:30
   local bA, bA2, bUU, u, Δ, Δ², uΔ, uA 
   bA = randn(sbA)
   bAA = basis2(bA)
   Δ = randn(size(bAA)) /  Diagonal(1:size(bAA, 2))
   Δ² = randn(size(bA)) /  Diagonal(1:size(bA, 2))
   uA = randn(size(bA)) /  Diagonal(1:size(bA, 2))
   uΔ = randn(size(bAA)) / Diagonal(1:size(bAA, 2))

   _Δ(t) = Δ + t * uΔ   
   _X(t) = bA + t * uA

   F(t) = dot(Δ², P4ML.pullback_evaluate(_Δ(t), basis2, _X(t)))
   dF(t) = begin
      val, pb = rrule(P4ML.pullback_evaluate, _Δ(t), basis2, _X(t))
      _, ∇_Δ, _, ∇_A = pb(Δ²)
      return dot(∇_Δ, uΔ) + dot(∇_A, uA)
   end
   F(0.0)
   dF(0.0)
   print_tf(@test fdtest(F, dF, 0.0; verbose=false))
end
println() 

=#

##

@info("Testing lux interface")

@info("Test consistency of lux and basis")
l_basis2 = P4ML.lux(basis2)
ps, st = Lux.setup(MersenneTwister(1234), l_basis2)
l_AA2, _ = l_basis2(bA, ps, st)
println_slim(@test l_AA2 ≈ basis2(bA))

println()
##


@info("Testing basic pushforward")

using ForwardDiff

for ntest = 1:10 
   local M, nX, spec, A, basis, AA1, AA2 
   M = rand(4:7)
   BO = rand(2:5)
   nX = rand(6:12)
   spec = generate_SO2_spec(BO, 2*M)
   A = randn(Float64, 2*M+1)
   ΔA = randn(length(A), nX)

   basis = SparseSymmProd(spec)
   AA1 = basis(A)
   ∂AA1 = ForwardDiff.jacobian(basis, A) * ΔA
   AA2, ∂AA2 = P4ML.pushforward_evaluate(basis, A, ΔA)
   print_tf( @test AA1 ≈ AA2 )
   print_tf( @test ∂AA1 ≈ ∂AA2 )
end 

##
