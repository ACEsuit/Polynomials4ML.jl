
@doc raw"""
`OrthPolyBasis1D3T:` defines a basis of polynomials in terms of a 3-term recursion, 
```math
\begin{aligned}
   P_1(x) &= A_1  \\
   P_2 &= A_2 x + B_2 \\
   P_{n} &= (A_n x + B_n) P_{n-1}(x) + C_n P_{n-2}(x)
\end{aligned}
```
Typically (but not necessarily) such bases are obtained by orthogonalizing the monomials with respect to a user-specified distribution, which
can be either continuous or discrete but must have a density function. See also 
* `legendre_basis`
* `chebyshev_basis`
* `jacobi_basis`
"""
struct OrthPolyBasis1D3T{N, T} <: AbstractP4MLBasis
   A::SVector{N, T}
   B::SVector{N, T}
   C::SVector{N, T}
end

function OrthPolyBasis1D3T(A::AbstractVector, B::AbstractVector, 
                           C::AbstractVector)
   N = length(A) 
   @assert N == length(B) == length(C)
   T = promote_type(eltype(A), eltype(B), eltype(C))
   return OrthPolyBasis1D3T(SVector{N, T}(A), 
                            SVector{N, T}(B), 
                            SVector{N, T}(C))
end

export OrthPolyBasis1D3T

natural_indices(basis::OrthPolyBasis1D3T) = 0:length(basis.A)-1

index(basis::OrthPolyBasis1D3T, m::Integer) = m + 1

Base.length(basis::OrthPolyBasis1D3T) = length(basis.A)

Base.show(io::IO, basis::OrthPolyBasis1D3T) = 
   print(io, "OrthPolyBasis1D3T(maxn = $(length(basis.A)))")

_valtype(basis::OrthPolyBasis1D3T{N, T1}, TX::Type{T2}) where {N, T1, T2} = 
            promote_type(T1, T2)

_generate_input(basis::OrthPolyBasis1D3T) = 2 * rand() - 1

function (T::Type{<: AbstractFloat})(basis::OrthPolyBasis1D3T) 
   return OrthPolyBasis1D3T(
      T.(basis.A), T.(basis.B), T.(basis.C))
end

# ----------------- CPU evaluation code 


function _evaluate!(P, dP, basis::OrthPolyBasis1D3T, X::BATCH)
   N = length(basis.A)
   nX = length(X) 
   WITHGRAD = !isnothing(dP)

   @inbounds begin 
      for i = 1:nX 
         P[i, 1] = basis.A[1]
         WITHGRAD && (dP[i, 1] = 0)
      end
      if N > 1
         for i = 1:nX 
            P[i, 2] = basis.A[2] * X[i] + basis.B[2]
            WITHGRAD && (dP[i, 2] = basis.A[2])
         end
         for n = 3:N
            an = basis.A[n]; bn = basis.B[n]; cn = basis.C[n]
            @simd ivdep for i = 1:nX 
               axb = muladd(an, X[i], bn)
               P[i, n] = muladd(axb, P[i, n-1], cn * P[i, n-2]) 
               WITHGRAD && (
                  q = muladd(cn,  dP[i, n-2], an * P[i, n-1]); 
                  dP[i, n] = muladd(axb, dP[i, n-1], q) 
                  )
            end
         end
      end
   end
   return nothing 
end


@kernel function _ka_evaluate!(P, dP, basis::OrthPolyBasis1D3T, X::BATCH)
   @uniform N = length(basis.A)
   @uniform nX = length(X) 
   @uniform WITHGRAD = !isnothing(dP)
   i = @index(Global)

   @inbounds begin 
      P[i, 1] = basis.A[1]
      WITHGRAD && (dP[i, 1] = 0)

      if N > 1
         P[i, 2] = basis.A[2] * X[i] + basis.B[2]
         WITHGRAD && (dP[i, 2] = basis.A[2])

         for n = 3:N
            an = basis.A[n]; bn = basis.B[n]; cn = basis.C[n]
            axb = muladd(an, X[i], bn)
            P[i, n] = muladd(axb, P[i, n-1], cn * P[i, n-2]) 
            WITHGRAD && (
               q = muladd(cn,  dP[i, n-2], an * P[i, n-1]); 
               dP[i, n] = muladd(axb, dP[i, n-1], q) 
               )
         end
      end
   end
   
   nothing 
end
