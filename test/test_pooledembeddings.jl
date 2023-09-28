

module PooledEmbed
using Polynomials4ML: AbstractPoly4MLBasis, PooledSparseProduct
using Polynomials4ML: @reqfields, POOL, TMP, _make_reqfields, META, evaluate
using Polynomials4ML
using StaticArrays
using ObjectPools: release!, acquire!
using ChainRulesCore
struct PooledEmebddings{NB, TB <: Tuple} <: AbstractPoly4MLBasis
   embeddings::TB
   pooling::PooledSparseProduct{NB}
   @reqfields()
end

function PooledEmebddings(embeddings, pooling)
   return PooledEmebddings(embeddings, pooling, _make_reqfields()...)
end

Base.length(basis::PooledEmebddings) = length(basis.pooling)

import Base.Cartesian: @nexprs

function _write_code_Bi_tup(NB)
   Bi_tup_str = "(B_1, "
   for i = 2:NB-1
      Bi_tup_str *= "B_$i, "
   end
   Bi_tup_str *= "B_$NB)"
   return Meta.parse(Bi_tup_str)
end

@generated function myevaluate(basis::PooledEmebddings{NB, TB}, X::Vector{SVector{3, T}}) where {NB, TB, T}
   @assert NB > 0
   B_tup = _write_code_Bi_tup(NB)
   quote
      @nexprs $NB i -> begin
         embed_i = basis.embeddings[i]
         B_i = evaluate(embed_i, X)
      end
      A = evaluate(basis.pooling, $B_tup)
      @nexprs $NB i -> release!(B_i)
      return A
   end
end

function ChainRulesCore.rrule(::typeof(myevaluate), basis::PooledEmebddings{NB, TB}, X::Vector{SVector{3, T}}) where {NB, TB, T}
   A = myevaluate(basis, X)
   return A, Δ -> _evaluate_pb(basis, Δ, X)
end

@generated function _evaluate_pb(basis::PooledEmebddings{NB, TB}, Δ, X) where {NB, TB}
   B_tup = _write_code_Bi_tup(NB)
   quote
      @nexprs $NB i -> begin
         embed_i = basis.embeddings[i]
         B_i = evaluate(embed_i, X)
      end
      A = evaluate(basis.pooling, $B_tup)
      pooling_out, pooling_pb = ChainRulesCore.rrule(evaluate, basis.pooling, $B_tup)
      @nexprs $NB i -> begin
         _, embed_pb_i = ChainRulesCore.rrule(evaluate, basis.embeddings[i], X)    
      end
      # pooling_pb : Vec -> (B_1, B_2, ..., B_NB)
      # embed_pb_i : B_i -> ∂X
      ∂X = similar(X)
      BBs = pooling_pb(Δ)[3]
      @show typeof(BBs)
      @nexprs $NB i -> begin
         ∂X_i = embed_pb_i(BBs[i])
         @show typeof(∂X_i)
         ∂X .+= ∂X_i[3]
      end
      @show ∂X
      return ∂X
   end
end

end # module PooledEmbed

using Polynomials4ML: legendre_basis, RYlmBasis, ScalarPoly4MLBasis, PooledSparseProduct, OrthPolyBasis1D3T
using StaticArrays, LinearAlgebra
using ObjectPools: acquire!, release!
using Polynomials4ML

function _generate_basis(; order=2, len = 50)
   NN = [ rand(5:10) for _ = 1:order ]
   spec = sort([ ntuple(t -> rand(1:NN[t]), order) for _ = 1:len])
   return PooledSparseProduct(spec)
end

function Polynomials4ML.evaluate!(Y::AbstractArray, basis::OrthPolyBasis1D3T, X::Vector{SVector{3, T}}) where T
   nX = length(X)
   R = acquire!(basis.pool, :R, (nX,), T)
   @simd ivdep for i = 1:nX
      R[i] = norm(X[i])
   end    
   evaluate!(Y, basis, R)
   release!(R)
	return Y
end
using ChainRulesCore
function ChainRulesCore.rrule(::typeof(Polynomials4ML.evaluate), basis::OrthPolyBasis1D3T, X::Vector{SVector{3, T}}) where T
   R = norm.(X)
   P, dP = evaluate_ed(basis, R)

   function pb(∂)
      ∂X = similar(X)
      for n = 1:size(dP, 2)
         @simd ivdep for a = 1:length(X)
            ∂X[a] += ∂[a, n] * dP[a, n] * X[a] / R[a]
         end
      end
      return (NoTangent(), NoTangent(), ∂X)
   end

  return Y, pb
end

Polynomials4ML._valtype(basis::OrthPolyBasis1D3T{T}, ::Type{<: StaticVector{3, S}}) where {T <: Real, S <: Real} = promote_type(T, S)

Rnl = legendre_basis(10)
Ylm = RYlmBasis(10)

pooling = _generate_basis()

X = [ @SVector(randn(3)) for i in 1:3 ]

embeddings = (Rnl, Ylm)
embed_and_pool = PooledEmbed.PooledEmebddings(embeddings, pooling)
Main.PooledEmbed.myevaluate(embed_and_pool, X)
using Zygote
out, pb = Zygote.pullback(X -> Main.PooledEmbed.myevaluate(embed_and_pool, X), X)

pb(out)