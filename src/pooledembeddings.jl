using Polynomials4ML: AbstractPoly4MLBasis, PooledSparseProduct

export PooledEmebddings

struct PooledEmebddings{NB, TB <: Tuple} <: AbstractPoly4MLBasis
   embeddings::TB
   pooling::PooledSparseProduct{NB}
   @reqfields()
end

function PooledEmebddings(embeddings, pooling)
   return PooledEmebddings(embeddings, pooling, _make_reqfields()...)
end

import Base.Cartesian: @nexprs

function _write_code_Bi_tup(NB)
   Bi_tup_str = "(B_1, "
   for i = 2:NB-1
      Bi_tup_str *= "B_$i, "
   end
   Bi_tup_str *= "B_$NB)"
   return Bi_tup_str
end

@generated function evaluate(basis::PooledEmebddings{NB, TB}, X)
   @assert NB > 0
   quote
      @nexprs $NB i -> begin
         embed_i = basis.embeddings[i]
         B_i = evaluate(embed_i, X)
      end
      B_tup = _write_code_Bi_tup(NB)
      A = basis.pooling($B_tup)
      @nexprs $NB i -> release!(B_i)
      return A
   end
end
