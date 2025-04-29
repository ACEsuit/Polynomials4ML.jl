
export MonoBasis


"""
Standard Monomials basis. This should very rarely be used. Possibly useful 
in combination with a transformation of the inputs, e.g. exponential. 
"""
struct MonoBasis{N} <: AbstractP4MLBasis
end

MonoBasis(N::Integer) = MonoBasis{N}()


# ----------------- interface functions 

natural_indices(basis::MonoBasis{N}) where {N} = 0:N

index(basis::MonoBasis, m::Integer) = m+1

Base.length(basis::MonoBasis{N}) where {N} = N+1

_valtype(basis::MonoBasis, T::Type{<: Number}) = T

_generate_input(basis::MonoBasis) = 2 * rand() - 1

# ----------------- main evaluation code 


function _evaluate!(P, dP, basis::MonoBasis{N}, X::BATCH)  where {N}
   nX = length(X)
   WITHGRAD = !isnothing(dP)

   @inbounds begin 
      @simd ivdep for i = 1:nX 
         P[i, 1] = 1 
         WITHGRAD && (dP[i, 1] = 0)
      end
      for n = 1:N 
         @simd ivdep for i = 1:nX
            P[i, n+1] = X[i] * P[i, n]
            WITHGRAD && (dP[i, n+1] = n * P[i, n])
         end
      end
   end
   return nothing 
end


@kernel function _ka_evaluate!(P, dP, basis::MonoBasis{N}, X::BATCH)  where {N}
   i = @index(Global)
   WITHGRAD = !isnothing(dP)

   @inbounds begin 
      P[i, 1] = 1 
      WITHGRAD && (dP[i, 1] = 0)
      for n = 1:N 
         P[i, n+1] = X[i] * P[i, n]
         WITHGRAD && (dP[i, n+1] = n * P[i, n])
      end
   end
   nothing
end
