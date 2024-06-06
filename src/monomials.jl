
export MonoBasis


"""
Standard Monomials basis. This should very rarely be used. Possibly useful 
in combination with a transformation of the inputs, e.g. exponential. 
"""
struct MonoBasis <: ScalarPoly4MLBasis
   N::Int
   # ----------------- metadata 
   @reqfields()
end

MonoBasis(N::Integer) = MonoBasis(N, _make_reqfields()...)


# ----------------- interface functions 



natural_indices(basis::MonoBasis) = 0:basis.N

index(basis::MonoBasis, m::Integer) = m+1

Base.length(basis::MonoBasis) = basis.N+1

_valtype(basis::MonoBasis, T::Type{<: Number}) = T

            
# ----------------- main evaluation code 

function evaluate!(P::AbstractArray, basis::MonoBasis, x::Number) 
   N = basis.N 
   @assert length(P) >= length(basis) 
   @inbounds P[1] = 1 
   @inbounds for n = 1:N 
      P[n+1] = x * P[n]
   end
   return P 
end

function evaluate!(P::AbstractArray, basis::MonoBasis, X::AbstractVector)
   N = basis.N 
   nX = length(X) 
   @assert size(P, 2) >= N+1
   @assert size(P, 1) >= nX 
   @inbounds begin 
      @simd ivdep for i = 1:nX 
         P[i, 1] = 1 
      end
      for n = 1:N 
         @simd ivdep for i = 1:nX
            P[i, n+1] = X[i] * P[i, n]
         end
      end
   end
   return P 
end

function evaluate_ed!(P::AbstractArray, dP::AbstractArray, basis::MonoBasis, x::Number) 
   N = basis.N 
   @assert length(P) >= length(basis) 
   @assert length(dP) >= length(basis) 
   @inbounds P[1] = 1 
   @inbounds dP[1] = 0 
   @inbounds for n = 1:N 
      P[n+1] = x * P[n] 
      dP[n+1] = n * P[n]
   end
   return P, dP 
end


function evaluate_ed!(P::AbstractArray, dP::AbstractArray, basis::MonoBasis, X::AbstractVector) 
   N = basis.N 
   nX = length(X)
   @assert size(P, 2) >= N+1
   @assert size(P, 1) >= nX
   @assert size(dP, 2) >= N+1
   @assert size(dP, 1) >= nX

   @inbounds begin 
      @simd ivdep for i = 1:nX 
         P[i, 1] = 1 
         dP[i, 1] = 0
      end
      for n = 1:N 
         @simd ivdep for i = 1:nX
            P[i, n+1] = X[i] * P[i, n]
            dP[i, n+1] = n * P[i, n]
         end
      end
   end
   return P, dP 
end

function evaluate_ed2!(P::AbstractArray, dP::AbstractArray, ddP::AbstractArray, basis::MonoBasis, x::Number) 
   N = basis.N 
   @assert length(P) >= length(basis) 
   @assert length(dP) >= length(basis) 
   @assert length(ddP) >= length(basis)

 
   @inbounds P[1] = 1 
   @inbounds dP[1] = 0 
   @inbounds ddP[1] = 0 

   @inbounds if N > 0 
      P[2] = x
      dP[2] = 1
      ddP[2] = 0 
      for n = 2:N 
         P[n+1] = x * P[n]
         dP[n+1] = n * P[n]
         ddP[n+1] = n*(n-1) * P[n-1]
      end
   end
   return P, dP, ddP 
end



function evaluate_ed2!(P::AbstractArray, dP::AbstractArray, ddP::AbstractArray, basis::MonoBasis, X::AbstractVector) 
   N = basis.N 
   nX = length(X)
   @assert size(P, 2) >= length(basis) 
   @assert size(dP, 2) >= length(basis) 
   @assert size(ddP, 2) >= length(basis)  
   @assert size(P, 1) >= nX 
   @assert size(dP, 1) >= nX 
   @assert size(ddP, 1) >= nX 

   @inbounds begin 
      @simd ivdep for i = 1:nX 
         P[i, 1] = 1 
         dP[i, 1] = 0
         ddP[i, 1] = 0
      end
   end

   @inbounds if N > 0
      @simd ivdep for i = 1:nX
         P[i, 2] = X[i]
         dP[i, 2] = 1
         ddP[i, 2] = 0
      end
      for n = 2:N 
         @simd ivdep for i = 1:nX
            P[i, n+1] = X[i] * P[i, n]
            dP[i, n+1] = n * P[i, n]
            ddP[i, n+1] = n*(n-1) * P[i, n-1]
         end
      end
   end

   return P, dP, ddP 
end
