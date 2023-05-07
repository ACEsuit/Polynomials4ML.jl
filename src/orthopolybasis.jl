using LoopVectorization


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
struct OrthPolyBasis1D3T{T} <: AbstractPoly4MLBasis
   # ----------------- the recursion coefficients
   A::Vector{T}
   B::Vector{T}
   C::Vector{T}
   # ----------------- used only for construction ...
   #                   but useful to have since it defines the notion of orth.
   meta::Dict{String, Any}
end

OrthPolyBasis1D3T(A, B, C) = OrthPolyBasis1D3T(A, B, C, Dict{String, Any}())

export OrthPolyBasis1D3T

natural_indices(basis::OrthPolyBasis1D3T) = 0:length(basis.A)-1

index(basis::OrthPolyBasis1D3T, m::Integer) = m + 1

Base.length(basis::OrthPolyBasis1D3T) = length(basis.A)


_valtype(basis::OrthPolyBasis1D3T{T1}, TX::Type{T2}) where {T1, T2} = 
            promote_type(T1, T2)         

# ----------------- main evaluation code 

# P must be a preallocated output vector of suitable length 
# x can be any object for which a polynomial could be defined, but normally 
#     a number. x cannot be an abstractvector since this would dispatch to a 
#     different method. 
function evaluate!(P, basis::OrthPolyBasis1D3T, x) 
   N = length(basis.A)
   @assert length(P) >= N 
   @inbounds P[1] = basis.A[1]
   if N > 1
      @inbounds P[2] = basis.A[2]*x + basis.B[2]
      @inbounds for n = 3:N
         # NB : fma seems to make no difference here 
         P[n] = (basis.A[n]*x + basis.B[n])*P[n-1] + basis.C[n]*P[n-2]
      end
   end
   return P
end


function evaluate_ed!(P, dP, basis::OrthPolyBasis1D3T, x)
   N = length(basis.A)
   @assert length(P) >= N 
   @inbounds begin 
      P[1] = basis.A[1]
      dP[1] = 0
      if N > 1
         P[2] = basis.A[2] * x + basis.B[2]
         dP[2] = basis.A[2]
         for n = 3:N
            axb = basis.A[n]*x + basis.B[n]
            P[n] = axb * P[n-1] + basis.C[n] * P[n-2]
            dP[n] = axb * dP[n-1] + basis.C[n] * dP[n-2] + basis.A[n] * P[n-1]
         end
      end
   end
   return P, dP 
end


function evaluate_ed2!(P, dP, ddP, basis::OrthPolyBasis1D3T, x)
   N = length(basis.A)
   @assert length(P) >= N 
   @inbounds begin 
      P[1] = basis.A[1]
      dP[1] = 0
      ddP[1] = 0
      if N > 1
         P[2] = basis.A[2] * x + basis.B[2]
         dP[2] = basis.A[2]
         ddP[2] = 0
         for n = 3:N
            axb = basis.A[n]*x + basis.B[n]
            P[n] = axb * P[n-1] + basis.C[n] * P[n-2]
            dP[n] = axb * dP[n-1] + basis.C[n] * dP[n-2] + basis.A[n] * P[n-1]
            ddP[n] = axb * ddP[n-1] + basis.C[n] * ddP[n-2] + 2 * basis.A[n] * dP[n-1]
         end
      end
   end
   return P, dP, ddP 
end


# P should be a matrix now and we will write basis(X[i]) into P[i, :]; 
# this is the format the optimizes memory access. 
function evaluate!(P, basis::OrthPolyBasis1D3T, X::AbstractVector) 
   N = length(basis.A)
   nX = length(X) 
   # ------- do the bounds checks here 
   @assert size(P, 2) >= N 
   @assert size(P, 1) >= nX
   # ---------------------------------

   @inbounds begin
      for i = 1:nX 
         P[i, 1] = basis.A[1]
      end
      if N > 1
         for i = 1:nX 
            P[i, 2] = basis.A[2] * X[i] + basis.B[2]
         end
         for n = 3:N    # TODO -> try @threads here 
            an = basis.A[n]; bn = basis.B[n]; cn = basis.C[n]
            @simd ivdep for i = 1:nX 
               p = muladd(X[i], an, bn)
               P[i, n] = muladd(p, P[i, n-1], cn * P[i, n-2])
            end
         end
      end
   end
   return P
end    



function evaluate_ed!(P, dP, basis::OrthPolyBasis1D3T, X::AbstractVector)
   N = length(basis.A)
   nX = length(X) 
   # ------- do the bounds checks here 
   @assert size(P, 2) >= N 
   @assert size(P, 1) >= nX
   @assert size(dP, 2) >= N 
   @assert size(dP, 1) >= nX
   # ---------------------------------

   @inbounds begin 
      for i = 1:nX 
         P[i, 1] = basis.A[1]
         dP[i, 1] = 0
      end
      if N > 1
         for i = 1:nX 
            P[i, 2] = basis.A[2] * X[i] + basis.B[2]
            dP[i, 2] = basis.A[2]
         end
         for n = 3:N
            an = basis.A[n]; bn = basis.B[n]; cn = basis.C[n]
            @simd ivdep for i = 1:nX 
               axb = muladd(an, X[i], bn)
               P[i, n] = muladd(axb, P[i, n-1], cn * P[i, n-2]) 
               q = muladd(cn,  dP[i, n-2], an * P[i, n-1])
               dP[i, n] = muladd(axb, dP[i, n-1], q)
            end
            # P[n] = axb * P[n-1] + basis.C[n] * P[n-2]
            # dP[n] = axb * dP[n-1] + basis.C[n] * dP[n-2] + basis.A[n] * P[n-1]
         end
      end
   end
   return P, dP 
end


function evaluate_ed2!(P, dP, ddP, basis::OrthPolyBasis1D3T, X::AbstractVector)
   N = length(basis.A)
   nX = length(X) 
   # ------- do the bounds checks here 
   @assert size(P, 2) >= N 
   @assert size(P, 1) >= nX
   @assert size(dP, 2) >= N 
   @assert size(dP, 1) >= nX
   @assert size(ddP, 2) >= N 
   @assert size(ddP, 1) >= nX
   # ---------------------------------

   @inbounds begin 
      for i = 1:nX 
         P[i, 1] = basis.A[1]
         dP[i, 1] = 0
         ddP[i, 1] = 0
      end
      if N > 1
         for i = 1:nX 
            P[i, 2] = basis.A[2] * X[i] + basis.B[2]
            dP[i, 2] = basis.A[2]
            ddP[i, 2] = 0
         end
         for n = 3:N
            an = basis.A[n]; bn = basis.B[n]; cn = basis.C[n]
            @simd ivdep for i = 1:nX 
               axb = muladd(an, X[i], bn)
               P[i, n] = muladd(axb, P[i, n-1], cn * P[i, n-2]) 
               q = muladd(cn,  dP[i, n-2], an * P[i, n-1])
               dP[i, n] = muladd(axb, dP[i, n-1], q)
               q1 = 2 * an * dP[i, n-1]
               q2 = muladd(cn, ddP[i, n-2], q1)
               ddP[i, n] = muladd(axb, ddP[i, n-1], q2)
            end
            # P[n] = axb * P[n-1] + basis.C[n] * P[n-2]
            # dP[n] = axb * dP[n-1] + basis.C[n] * dP[n-2] + basis.A[n] * P[n-1]
            # ddP[n] = axb * ddP[n-1] + basis.C[n] * ddP[n-2] + 2 * basis.A[n] * dP[n-1]
            end
      end
   end
   return P, dP, ddP 
end


# ------------------   rrules 

# 
# ∂_xa ( ∂P : P ) = ∑_ij ∂_xa ( ∂P_ij * P_ij ) 
#                 = ∑_ij ∂P_ij * ∂_xa ( P_ij )
#                 = ∑_ij ∂P_ij * dP_ij δ_ia
#
function rrule_evaluate!(P, basis::OrthPolyBasis1D3T, X::AbstractVector)
   nX = length(X) 
   dP = similar(P)
   evaluate_ed!(P, dP, basis, X)

   function pb(∂P)
      ∂X = zeros(nX)
      for j = 1:length(basis) 
         for i = 1:nX 
            ∂X[i] += ∂P[i, j] * dP[i, j]
         end
      end
      return ∂X 
   end

   return P, pb 
end
