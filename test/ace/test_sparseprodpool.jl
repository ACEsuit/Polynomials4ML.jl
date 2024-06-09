
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
nX = 64

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

@warn("order = 1 tests currently fail in an unexplained way")

for ntest = 1:30
   local bBB, bA2, u, basis, nX 
   order = mod1(ntest, 3)+1
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

@info("Testing pb_pb_evaluate for PooledSparseProduct")
import ChainRulesCore: rrule, NoTangent

for ntest = 1:20 
   local basis, val, pb, bBB, A 
   ORDER = mod1(ntest, 3)+1
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

   val2, pb2 = rrule(P4ML.pullback_evaluate, ∂A, basis, bBB)
   @test val2 == ∂_BB

   ∂2 = ntuple(i -> randn(size(∂_BB[i])), length(∂_BB))
   bUU = _generate_input(basis; nX = size(bBB[1], 1))
   _BB(t) = ntuple(i -> bBB[i] + t * bUU[i], ORDER)
   bV = randn(size(∂A))
   _∂A(t) = ∂A + t * bV

   F(t) = begin
      ∂_BB = P4ML.pullback_evaluate(_∂A(t), basis, _BB(t))
      return sum(dot(∂2[i], ∂_BB[i]) for i = 1:length(∂_BB))
   end
   dF(t) = begin
      val, pb = rrule(P4ML.pullback_evaluate, ∂A, basis, _BB(t))
      _, ∂_∂A, _, ∂2_BB = pb(∂2)
      return dot(∂_∂A, bV) + sum(dot(bUU[i], ∂2_BB[i]) for i = 1:ORDER)
   end

   print_tf(@test all( fdtest(F, dF, 0.0; verbose=false) ))
end
println()

##

ORDER = 2
basis = _generate_basis(;order = ORDER, len=300)
bBB = _generate_input(basis)
∂A = randn(length(basis))
∂2 = ntuple(i -> randn(size(bBB[i])), length(bBB))

@btime Polynomials4ML.pullback_evaluate($∂A, $basis, $bBB)
@btime Polynomials4ML.pb_pb_evaluate($∂2, $∂A, $basis, $bBB);

##

using ForwardDiff: Dual, extract_derivative 

function auto_pb_pb(∂BB, ∂A, basis, BB) 
   # φ = ∂BB ⋅ pullback(∂A, basis, BB)
   #   = (∂bBB ⋅ ∇_BB) (∂A ⋅ evaluate(basis, BB))
   # ∇_∂A φ = (∂BB ⋅ ∇_BB) evaluate(basis, BB)
   # ∇_BB φ = (∂BB ⋅ ∇_BB) ∇_BB (∂A ⋅ evaluate(basis, BB))
   #        = (∂BB ⋅ ∇_BB) pullback(∂A, basis, BB)
   d = Dual{Float64}(0.0, 1.0)
   BB_d = ntuple(i -> BB[i] .+ d .* ∂BB[i], length(BB))
   @no_escape begin 
      A_d = @withalloc evaluate!(basis, BB_d)
      ∂BB_d = @withalloc pullback_evaluate!(∂A, basis, BB_d)
      ∇_∂A = extract_derivative.(Float64, A_d)
      ∇_BB = ntuple(i -> extract_derivative.(Float64, ∂BB_d[i]), length(∂BB_d))
   end
   return ∇_∂A, ∇_BB
end

using Bumper, WithAlloc

function auto_pb_pb!(∇_∂A, ∇_BB, ∂BB, ∂A, basis, BB) 
   @assert all(eltype(BB[i]) == eltype(BB[1]) for i = 2:length(BB))
   @no_escape begin 
      T = eltype(BB[1])
      d = Dual{T}(zero(T), one(T))
      TD = typeof(d)
      B1 = BB[1] 
      B2 = BB[2] 
      B1_d = @alloc(TD, size(B1)...)
      B2_d = @alloc(TD, size(B2)...)
      for t = 1:length(B1)
         B1_d[t] = B1[t] + d * ∂BB[1][t]
      end
      for t = 1:length(B2)
         B2_d[t] = B2[t] + d * ∂BB[2][t]
      end
      BB_d = (B1_d, B2_d)
      A_d = @withalloc evaluate!(basis, BB_d)
      ∂BB_d = @withalloc pullback_evaluate!(∂A, basis, BB_d)
      for i = 1:length(A_d)
         ∇_∂A[i] = extract_derivative(T, A_d[i])
      end
      for i = 1:length(∂BB_d)
         for j = 1:length(∂BB_d[i])
            ∇_BB[i][j] = extract_derivative(T, ∂BB_d[i][j])
         end
      end
   end
   return ∇_∂A, ∇_BB
end


∇_∂A, ∇_BB = auto_pb_pb(∂2, ∂A, basis, bBB);
auto_pb_pb!(∇_∂A2, ∇_BB2, ∂2, ∂A, basis, bBB)
@btime auto_pb_pb($∂2, $∂A, $basis, $bBB);
@btime auto_pb_pb!($∇_∂A2, $∇_BB2, $∂2, $∂A, $basis, $bBB);

##

∇_∂A1, ∇_BB1 = auto_pb_pb(∂2, ∂A, basis, bBB)
∇_∂A2, ∇_BB2 = P4ML.pb_pb_evaluate(∂2, ∂A, basis, bBB)
∇_∂A3 = deepcopy(∇_∂A2); ∇_BB3 = deepcopy(∇_BB2)
auto_pb_pb!(∇_∂A3, ∇_BB3, ∂2, ∂A, basis, bBB)
∇_∂A1 ≈ ∇_∂A2 ≈ ∇_∂A3
all(∇_BB1 .≈ ∇_BB2 .≈ ∇_BB3)

## 
@info("Testing pushforward for PooledSparseProduct")

using ForwardDiff

function _rand_input1_pfwd(basis::PooledSparseProduct{ORDER}; 
                      nX = rand(7:12)) where {ORDER} 
   NN = [ maximum(b[i] for b in basis.spec) for i = 1:ORDER ]
   BB = ntuple(i -> randn(nX, NN[i]), ORDER)
   ΔBB = ntuple(i -> randn(nX, NN[i]), ORDER)
   return BB, ΔBB
end

function fwddiff1_pfwd(basis::PooledSparseProduct{NB}, BB, ΔBB) where {NB}
   A1 = basis(BB)
   sub_i(t, ti, i) = ntuple(a -> a == i ? ti : t[a], length(t))
   ∂A1_i = [  ForwardDiff.jacobian(B -> basis(sub_i(BB, B, i)), BB[i])
              for i = 1:NB ]
   ∂A1 = sum(∂A1_i[i] * ΔBB[i] for i = 1:NB)            
   return A1, ∂A1   
end

function fwddiff_pfwd(basis::PooledSparseProduct{NB}, BB, ΔBB) where {NB}
   nX = size(BB[1], 1)
   Aj_∂Aj = [ fwddiff1_pfwd(basis, 
                         ntuple(t ->  BB[t][j,:], NB), 
                         ntuple(t -> ΔBB[t][j,:], NB), ) 
               for j = 1:nX ] 
   Aj = [ x[1] for x in Aj_∂Aj ] 
   ∂Aj = [ x[2] for x in Aj_∂Aj ] 
   A = sum(Aj) 
   ∂A = reduce(hcat, ∂Aj)               
   return A, ∂A
end


for ntest = 1:10 
   local order, basis, BB, ΔBB, A1, ∂A1, A2, ∂A2 
   order = rand(2:4)
   basis = _generate_basis(; order=order)
   BB, ΔBB = _rand_input1_pfwd(basis)
   A1, ∂A1 = fwddiff_pfwd(basis, BB, ΔBB)
   A2, ∂A2 = P4ML.pushforward_evaluate(basis, BB, ΔBB)
   print_tf(@test A2 ≈ A1)
   print_tf(@test ∂A2 ≈ ∂A1)
end

##

# # quick performance and allocation check
# using ObjectPools: unwrap 
# order = 3
# basis = _generate_basis(; order=order)
# BB, ΔBB = _rand_input1_pfwd(basis)
# A, ∂A = P4ML.pfwd_evaluate(basis, BB, ΔBB)
# @btime Polynomials4ML.pfwd_evaluate!($(unwrap(A)), $(unwrap(∂A)), $basis, $BB, $ΔBB)


