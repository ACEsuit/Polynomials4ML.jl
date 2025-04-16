export ChebBasis

@doc raw"""
`ChebBasis(N)`: 

Chebyshev polynomials up to degree `N-1` (inclusive). i.e  basis with length `N`. The basis is ordered as 
```math
[1, x, 2x^2-1, 4x^3-3x, ..., 2xT_{N-1}(x)-T_{N-2}(x)]
```
where `x` is input variable. 

The differences between `ChebBasis` and `chebyshev_basis` is that `ChebBasis` 
computes the basis on the fly when it is compiled and it does not store the 
recursion coefficients as in `chebyshev_basis`. There might be a small 
performance benefit from this. 

Secondly, `ChebBasis` and `chebyshev_basis` use different normalization.
"""
struct ChebBasis{N} <: AbstractP4MLBasis where {N} 
   @reqfields
end

ChebBasis(N::Integer) = ChebBasis{N}(_make_reqfields()...)

Base.length(basis::ChebBasis{N}) where {N} = N

natural_indices(basis::ChebBasis) = 0:length(basis)-1

_valtype(basis::ChebBasis, T::Type{<:Real}) = T

_generate_input(basis::ChebBasis) = 2 * rand() - 1 

_ref_evaluate(basis::ChebBasis, x::Real) = 
      [ cos( (n-1) * acos(x)) for n = 1:length(basis) ]


# --------------------------------------------------------- 
# CPU SIMD kernel 
# 

function _evaluate!(P, dP, 
                    basis::ChebBasis{N},
                    x::AbstractVector{<:Real}) where {N} 
   nX = length(x)
   WITHGRAD = !isnothing(dP)

   @inbounds begin
      @simd ivdep for i = 1:nX
         P[i, 1] = 1
         WITHGRAD && (dP[i, 1] = 0)
         P[i, 2] = x[i]
         WITHGRAD && (dP[i, 2] = 1)
      end

      for k = 3:N
         @simd ivdep for i = 1:nX
            P[i, k] = 2 * x[i] * P[i, k-1] - P[i, k-2]
            WITHGRAD && (dP[i, k] = 2 * P[i, k-1] + 2 * x[i] * dP[i, k-1] - dP[i, k-2])
         end
      end
   end
   return nothing 
end


# --------------------------------------------------------- 
# KernelAbstractions kernel
# 

@kernel function _ka_evaluate!(P, dP, basis::ChebBasis{N}, x::AbstractVector{T}
         ) where {T, N}
            
   i = @index(Global)
   @uniform WITHGRAD = !isnothing(dP)

   @inbounds begin
      P[i, 1] = 1
      WITHGRAD && (dP[i, 1] = 0)
      if N > 1
         P[i, 2] = x[i]
         WITHGRAD && (dP[i, 2] = 1)
      end 
      for n = 3:N 
         P[i, n] = 2 * x[i] * P[i, n-1] - P[i, n-2]
         WITHGRAD && ( 
            dP[i, n] = 2 * P[i, n-1] + 2 * x[i] * dP[i, n-1] - dP[i, n-2] )
      end
   end
end
