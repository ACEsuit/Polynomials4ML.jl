export ChebBasis

@doc raw"""
`ChebBasis(N)`: 

Chebyshev polynomials up to degree `N-1` (inclusive). i.e  basis with length `N`. The basis is ordered as 
```math
[1, x, 2x^2-1, 4x^3-3x, ..., 2xT_{N-1}(x)-T_{N-2}(x)]
```
where `x` is input variable. 

The differences between `ChebBasis` and `chebyshev_basis` is that `ChebBasis` computes the basis on the go when it is compiled and it does not store the recursion coefficients as in `chebyshev_basis`.

Warning: `ChebBasis` and `chebyshev_basis` have different normalization.
"""
struct ChebBasis <: ScalarPoly4MLBasis
   N::Int
   @reqfields
end

ChebBasis(N::Integer) = ChebBasis(N, _make_reqfields()...)

Base.length(basis::ChebBasis) = basis.N

natural_indices(basis::ChebBasis) = 0:length(basis)-1

_valtype(basis::ChebBasis, T::Type{<:Real}) = T


function evaluate!(P::AbstractVector, basis::ChebBasis, x::Real)
   N = basis.N
   @assert N >= 2
   @assert length(P) >= length(basis) # N

   P[1] = 1
   P[2] = x
   for k = 3:N
      @inbounds P[k] = 2 * x * P[k-1] - P[k-2]
   end
   return P
end



function evaluate!(P::AbstractMatrix, basis::ChebBasis,
   x::AbstractVector{<:Real})
   N = basis.N
   nX = length(x)
   @assert N >= 2
   @assert size(P, 2) >= length(basis) # N
   @assert size(P, 1) >= nX

   @inbounds begin
      @simd ivdep for i = 1:nX
         P[i, 1] = 1
         P[i, 2] = x[i]
      end

      for k = 3:N
         @simd ivdep for i = 1:nX
            P[i, k] = 2 * x[i] * P[i, k-1] - P[i, k-2]
         end
      end
   end
   return P
end

function evaluate_ed!(P::AbstractVector, dP::AbstractVector,
   basis::ChebBasis, x::Real)
   N = basis.N
   nX = length(x)
   @assert N >= 2
   @assert length(P) >= length(basis)
   @assert length(dP) >= length(basis)

   @inbounds begin
      P[1] = 1
      dP[1] = 0
      P[2] = x
      dP[2] = 1
      for k = 3:N
         P[k] = 2 * x * P[k-1] - P[k-2]
         dP[k] = 2 * P[k-1] + 2 * x * dP[k-1] - dP[k-2]
      end
   end
   return P, dP
end


function evaluate_ed!(P::AbstractMatrix, dP::AbstractMatrix, basis::ChebBasis,
   x::AbstractVector{<:Real})
   N = basis.N
   nX = length(x)
   @assert N >= 2
   @assert size(P, 2) >= length(basis) # N
   @assert size(P, 1) >= nX
   @assert size(dP, 2) >= length(basis) # N
   @assert size(dP, 1) >= nX

   @inbounds begin
      @simd ivdep for i = 1:nX
         P[i, 1] = 1
         dP[i, 1] = 0
         P[i, 2] = x[i]
         dP[i, 2] = 1
      end

      for k = 3:N
         @simd ivdep for i = 1:nX
            P[i, k] = 2 * x[i] * P[i, k-1] - P[i, k-2]
            dP[i, k] = 2 * P[i, k-1] + 2 * x[i] * dP[i, k-1] - dP[i, k-2]
         end
      end
   end
   return P, dP
end


function evaluate_ed2!(P::AbstractVector, dP::AbstractVector, ddP::AbstractVector,
   basis::ChebBasis, x::Real)
   N = basis.N
   @assert N >= 2
   @assert length(P) >= length(basis) # N
   @assert length(dP) >= length(basis) # N
   @assert length(ddP) >= length(basis) # N

   @inbounds begin
      P[1] = 1
      P[2] = x
      dP[1] = 0
      dP[2] = 1
      ddP[1] = 0
      ddP[2] = 0

      for k = 3:N
         P[k] = 2 * x * P[k-1] - P[k-2]
         dP[k] = 2 * P[k-1] + 2 * x * dP[k-1] - dP[k-2]
         ddP[k] = 2 * dP[k-1] + 2 * dP[k-1] + 2 * x * ddP[k-1] - ddP[k-2]
      end
   end
   return P, dP, ddP
end



function evaluate_ed2!(P::AbstractMatrix, dP::AbstractMatrix, ddP::AbstractMatrix, basis::ChebBasis,
   x::AbstractVector{<:Real})
   N = basis.N
   nX = length(x)
   @assert N >= 2
   @assert size(P, 2) >= length(basis) # N
   @assert size(P, 1) >= nX
   @assert size(dP, 2) >= length(basis) # N
   @assert size(dP, 1) >= nX
   @assert size(ddP, 2) >= length(basis) # N
   @assert size(ddP, 1) >= nX

   @inbounds begin
      @simd ivdep for i = 1:nX
         P[i, 1] = 1
         P[i, 2] = x[i]
         dP[i, 1] = 0
         dP[i, 2] = 1
         ddP[i, 1] = 0
         ddP[i, 2] = 0
      end

      for k = 3:N
         @simd ivdep for i = 1:nX
            P[i, k] = 2 * x[i] * P[i, k-1] - P[i, k-2]
            dP[i, k] = 2 * P[i, k-1] + 2 * x[i] * dP[i, k-1] - dP[i, k-2]
            ddP[i, k] = 2 * dP[i, k-1] + 2 * dP[i, k-1] + 2 * x[i] * ddP[i, k-1] - ddP[i, k-2]
         end
      end
   end
   return P, dP, ddP
end
