
using BenchmarkTools, Test, Polynomials4ML
using Polynomials4ML: PooledSparseProduct, evaluate, evaluate!
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

function _rand_input1(basis::PooledSparseProduct{ORDER}) where {ORDER} 
   NN = [ maximum(b[i] for b in basis.spec) for i = 1:ORDER ]
   BB = ntuple(i -> randn(NN[i]), ORDER)
end

function _rand_input(basis::PooledSparseProduct{ORDER}; nX = rand(5:15)) where {ORDER} 
   NN = [ maximum(b[i] for b in basis.spec) for i = 1:ORDER ]
   BB = ntuple(i -> randn(nX, NN[i]), ORDER)
end

##

@info("Test evaluation with a single input (no pooling)")

for ntest = 1:30
   local BB, A1, A2, basis

   order = mod1(ntest, 4)
   basis = _generate_basis(; order=order)
   BB = _rand_input1(basis)
   A1 = test_evaluate(basis, BB)
   A2 = evaluate(basis, BB)
   print_tf(@test A1 ≈ A2)
end
println()

## 

@info("Test pooling of multiple inputs")
nX = 64

for ntest = 1:30 
   local bBB, bA1, bA2, bA3, basis 

   order = mod1(ntest, 4)
   basis = _generate_basis(; order=order)
   bBB = _rand_input(basis)
   bA1 = test_evaluate(basis, bBB)
   bA2 = evaluate(basis, bBB)
   bA3 = copy(bA2)
   evaluate!(bA3, basis, bBB)

   print_tf(@test bA1 ≈ bA2 ≈ bA3)
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

@info("     _pb_prod_grad")
for ORDER = 1:5
   local val, pb, u 
   @info("order = $ORDER")
   pb_prodgrad = P4ML._pb_prod_grad
   b = rand(SVector{ORDER,Float64})
   g = prodgrad(b.data, Val(ORDER))
   ∂ = @SVector randn(length(g))
   val, pb = pb_prodgrad(∂.data, b.data, Val(ORDER))
   @test all(val .≈ g)
   u = randn(SVector{ORDER,Float64}) 
   println_slim(@test  fdtest( b -> sum(u .* prodgrad(tuple(b...), Val(ORDER))), 
            b -> [ pb_prodgrad(u.data, tuple(b...), Val(ORDER))[2]... ],
            [b...], verbose = false ) |> all )
end

##

@info("Testing _rrule_evalpool")
using LinearAlgebra: dot

for ntest = 1:30
   local bBB, bA2, u, basis, nX 
   order = mod1(ntest, 4)
   basis = _generate_basis(; order=order)
   bBB = _rand_input(basis)
   nX = size(bBB[1], 1)
   bUU = _rand_input(basis; nX = nX) # same shape and type as bBB 
   _BB(t) = ntuple(i -> bBB[i] + t * bUU[i], order)
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

## 

@info("Testing _pb_pb_evaluate for PooledSparseProduct")
import ChainRulesCore: rrule, NoTangent

for ntest = 1:20 
   local basis, val, pb 
   ORDER = mod1(ntest, 3)+1
   basis = _generate_basis(;order = ORDER)
   bBB = _rand_input(basis)
   ∂A = randn(length(basis))

   A = evaluate(basis, bBB)
   val, pb = rrule(evaluate, basis, bBB)
   nt1, nt2, ∂_BB = pb(∂A)

   @test val ≈ A
   @test nt1 isa NoTangent && nt2 isa NoTangent
   @test ∂_BB isa NTuple{ORDER, <: AbstractMatrix}
   @test all(size(∂_BB[i]) == size(bBB[i]) for i = 1:length(bBB))

   val2, pb2 = rrule(P4ML._pullback_evaluate, ∂A, basis, bBB)
   @test val2 == ∂_BB

   ∂2 = ntuple(i -> randn(size(∂_BB[i])), length(∂_BB))
   bUU = _rand_input(basis; nX = size(bBB[1], 1))
   _BB(t) = ntuple(i -> bBB[i] + t * bUU[i], ORDER)
   bV = randn(size(∂A))
   _∂A(t) = ∂A + t * bV

   F(t) = begin
      ∂_BB = P4ML._pullback_evaluate(_∂A(t), basis, _BB(t))
      return sum(dot(∂2[i], ∂_BB[i]) for i = 1:length(∂_BB))
   end
   dF(t) = begin
      val, pb = rrule(P4ML._pullback_evaluate, ∂A, basis, _BB(t))
      _, ∂_∂A, _, ∂2_BB = pb(∂2)
      return dot(∂_∂A, bV) + sum(dot(bUU[i], ∂2_BB[i]) for i = 1:ORDER)
   end

   print_tf(@test all( fdtest(F, dF, 0.0; verbose=false) ))
end
println()

