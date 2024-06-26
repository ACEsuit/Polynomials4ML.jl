
using BenchmarkTools, Test, Polynomials4ML, ChainRulesCore
using Polynomials4ML: PooledSparseProduct, evaluate, evaluate!, 
         _generate_input, _generate_input_1
using Polynomials4ML.Testing: test_withalloc
using ACEbase.Testing: fdtest, println_slim, print_tf 

test_evaluate(basis::PooledSparseProduct, BB::Tuple{Vararg{AbstractVector}}) =
   [prod(BB[j][basis.spec[i][j]] for j = 1:length(BB))
    for i = 1:length(basis)]

test_evaluate(basis::PooledSparseProduct, BB::Tuple{Vararg{AbstractMatrix}}) =
   sum(test_evaluate(basis, ntuple(i -> BB[i][j, :], length(BB)))
       for j = 1:size(BB[1], 1))

P4ML = Polynomials4ML

##

function _generate_basis(; order=3, len = 50)
   NN = [ rand(10:30) for _ = 1:order ]
   spec = sort([ ntuple(t -> rand(1:NN[t]), order) for _ = 1:len])
   return PooledSparseProduct(spec)
end


##

@info("Test evaluation with a single input (no pooling)")

for ntest = 1:30
   local BB, A1, A2, basis

   order = mod1(ntest, 4)
   basis = _generate_basis(; order=order)
   BB = _generate_input_1(basis)
   A1 = test_evaluate(basis, BB)
   A2 = evaluate(basis, BB)
   print_tf(@test A1 ≈ A2)
end
println()

## 

@info("Test pooling of multiple inputs")
nX = 17

for ntest = 1:30 
   local bBB, bA1, bA2, bA3, basis 

   order = mod1(ntest, 4)
   basis = _generate_basis(; order=order)
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
basis = _generate_basis(; order=2)
BB = _generate_input_1(basis)
bBB = _generate_input(basis)
test_withalloc(basis; batch=false)


##


@info("Testing rrule")
using LinearAlgebra: dot

for ntest = 1:30
   local bBB, bA2, u, basis, nX 
   order = mod1(ntest, 4)
   basis = _generate_basis(; order=order)
   bBB = _generate_input(basis)
   nX = size(bBB[1], 1)
   bUU = _generate_input(basis; nX = nX) # same shape and type as bBB 
   _BB(t) = ntuple(i -> bBB[i] + t * bUU[i], order)
   bA2 = evaluate(basis, bBB)
   u = randn(size(bA2))
   F(t) = dot(u, evaluate(basis, _BB(t)))
   dF(t) = begin
      val, pb = rrule(evaluate, basis, _BB(t))
      ∂BB = pb(u)[3]
      return sum(dot(∂BB[i], bUU[i]) for i = 1:length(bUU))
   end
   print_tf(@test fdtest(F, dF, 0.0; verbose=false))
end
println()

## 

@info("Testing pullback2 for PooledSparseProduct")
import ChainRulesCore: rrule, NoTangent

for ntest = 1:20
   local basis, val, pb, bBB, A 
   ORDER = mod1(ntest, 4)
   basis = _generate_basis(;order = ORDER)
   bBB = _generate_input(basis)
   ∂A = randn(length(basis))

   A = evaluate(basis, bBB)
   val, pb = rrule(evaluate, basis, bBB)
   nt1, nt2, ∂_BB = pb(∂A)

   @test val ≈ A
   @test nt1 isa NoTangent && nt2 isa NoTangent
   @test ∂_BB isa NTuple{ORDER, <: AbstractMatrix}
   @test all(size(∂_BB[i]) == size(bBB[i]) for i = 1:length(bBB))

   val2, pb2 = rrule(P4ML.pullback, ∂A, basis, bBB)
   @test val2 == ∂_BB

   ∂2 = ntuple(i -> randn(size(∂_BB[i])), length(∂_BB))
   bUU = _generate_input(basis; nX = size(bBB[1], 1))
   _BB(t) = ntuple(i -> bBB[i] + t * bUU[i], ORDER)
   bV = randn(size(∂A))
   _∂A(t) = ∂A + t * bV

   F(t) = begin
      ∂_BB = P4ML.pullback(_∂A(t), basis, _BB(t))
      return sum(dot(∂2[i], ∂_BB[i]) for i = 1:length(∂_BB))
   end
   dF(t) = begin
      val, pb = rrule(P4ML.pullback, ∂A, basis, _BB(t))
      _, ∂_∂A, _, ∂2_BB = pb(∂2)
      return dot(∂_∂A, bV) + sum(dot(bUU[i], ∂2_BB[i]) for i = 1:ORDER)
   end

   print_tf(@test fdtest(F, dF, 0.0; verbose=false) )
end
println()


## 
@info("Testing pushforward for PooledSparseProduct")

using Polynomials4ML: pushforward

for ntest = 1:20 
   local order, basis, BB, ΔBB, A1, ∂A1, A2, ∂A2 
   order = mod1(ntest, 4)
   basis = _generate_basis(; order=order)
   BB = _generate_input(basis) 
   ΔBB = ntuple(i -> randn(Float64, size(BB[i])), order) 
   _BB(t) = ntuple(i -> BB[i] + t * ΔBB[i], order)
   U = randn(length(basis)) ./ (1:length(basis))
   A, ∂A = pushforward(basis, BB, ΔBB)
   print_tf(@test A ≈ basis(BB))
   F(t) = dot(U, evaluate(basis, _BB(t)))
   dF(t) = dot(U, pushforward(basis, _BB(t), ΔBB)[2])
   print_tf(@test fdtest(F, dF, 0.0; verbose=false))
end
println() 
