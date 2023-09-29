export PooledEmbeddings

using ChainRulesCore: NoTangent
using Polynomials4ML: PooledSparseProduct, AbstractPoly4MLBasis

import ChainRulesCore: rrule
import Base.Cartesian: @nexprs

struct PooledEmbeddings{NB, TB <: Tuple} <: AbstractPoly4MLBasis
   embeddings::TB
   pooling::PooledSparseProduct{NB}
   @reqfields()
end

function PooledEmbeddings(embeddings, pooling)
   return PooledEmbeddings(embeddings, pooling, _make_reqfields()...)
end

Base.length(basis::PooledEmbeddings) = length(basis.pooling)

function _write_code_Bi_tup(NB)
   Bi_tup_str = "(B_1, "
   for i = 2:NB-1
      Bi_tup_str *= "B_$i, "
   end
   Bi_tup_str *= "B_$NB)"
   return Meta.parse(Bi_tup_str)
end

# TODO: generalize to any X
@generated function evaluate(basis::PooledEmbeddings{NB, TB}, X) where {NB, TB}
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

function ChainRulesCore.rrule(::typeof(evaluate), basis::PooledEmbeddings{NB, TB}, X) where {NB, TB}
   A = evaluate(basis, X)
   return A, Δ -> _evaluate_pb(basis, Δ, X)
end

@generated function _evaluate_pb(basis::PooledEmbeddings{NB, TB}, Δ, X) where {NB, TB}
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
      # ∂X = similar(X)
      ∂BBs = pooling_pb(Δ)[3]
      # writes and accumulate to ∂X
      ∂X = embed_pb_1(∂BBs[1])[3]
      @nexprs $(NB-1) i -> begin
         ∂X_{i+1} = embed_pb_{i+1}(∂BBs[i+1])
         ∂X .+= ∂X_{i+1}[3]
      end
      return (NoTangent(), NoTangent(), ∂X)
   end
end


# lux integration to prevent it dispatch to wrong methods, ObjectPools is not required here since it is done internally in each embedding
function evaluate(l::Polynomials4ML.PolyLuxLayer{PooledEmbeddings{NB, TB}}, X, ps, st) where {NB, TB}
   return evaluate(l.basis, X), st
end