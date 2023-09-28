export PooledEmebddings

using ChainRulesCore: NoTangent
using Polynomials4ML: PooledSparseProduct, AbstractPoly4MLBasis

import ChainRulesCore: rrule
import Base.Cartesian: @nexprs

struct PooledEmebddings{NB, TB <: Tuple} <: AbstractPoly4MLBasis
   embeddings::TB
   pooling::PooledSparseProduct{NB}
   @reqfields()
end

function PooledEmebddings(embeddings, pooling)
   return PooledEmebddings(embeddings, pooling, _make_reqfields()...)
end

Base.length(basis::PooledEmebddings) = length(basis.pooling)

function _write_code_Bi_tup(NB)
   Bi_tup_str = "(B_1, "
   for i = 2:NB-1
      Bi_tup_str *= "B_$i, "
   end
   Bi_tup_str *= "B_$NB)"
   return Meta.parse(Bi_tup_str)
end

# TODO: generalize to any X
@generated function evaluate(basis::PooledEmebddings{NB, TB}, X::Vector{SVector{3, T}}) where {NB, TB, T}
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

function ChainRulesCore.rrule(::typeof(evaluate), basis::PooledEmebddings{NB, TB}, X::Vector{SVector{3, T}}) where {NB, TB, T}
   @show "Calling correct rrule"
   A = evaluate(basis, X)
   return A, Δ -> _evaluate_pb(basis, Δ, X)
end

@generated function _evaluate_pb(basis::PooledEmebddings{NB, TB}, Δ, X) where {NB, TB}
   B_tup = _write_code_Bi_tup(NB)
   quote
      # evaluate
      @nexprs $NB i -> begin
         embed_i = basis.embeddings[i]
         B_i = evaluate(embed_i, X)
      end
      A = evaluate(basis.pooling, $B_tup)

      # pooling
      pooling_out, pooling_pb = ChainRulesCore.rrule(evaluate, basis.pooling, $B_tup)

      # get the embedding pullbacks
      @nexprs $NB i -> begin
         _, embed_pb_i = ChainRulesCore.rrule(evaluate, basis.embeddings[i], X)    
      end
      
      # pooling_pb : Vec -> (B_1, B_2, ..., B_NB)
      # embed_pb_i : B_i -> ∂X
      # use 3 since the interface must return (NoTangent(), NoTangent(), ∂)
      ∂X = zero(X)
      ∂BBs = pooling_pb(Δ)[3]
      # writes and accumulate to ∂X
      @nexprs $NB i -> begin
         ∂X_i = embed_pb_i(∂BBs[i])
         ∂X .+= ∂X_i[3]
      end
      @show ∂X
      return (NoTangent(), NoTangent(), ∂X)
   end
end