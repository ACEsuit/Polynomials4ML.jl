using HyperDualNumbers: Hyper

export RTrigBasis

"""
`RTrigBasis(N)`: 

Real trigonometric polynomials up to degree `N` (inclusive). The basis is ordered as 
```
[1, cos(θ), sin(θ), cos(2θ), sin(2θ), ..., cos(Nθ), sin(Nθ) ]
```
where `θ` is input variable. 
"""
struct RTrigBasis <: ScalarPoly4MLBasis
   N::Int
   @reqfields
end

RTrigBasis(N::Integer) = RTrigBasis(N, _make_reqfields()...)

Base.length(basis::RTrigBasis) = 2 * basis.N + 1

function natural_indices(basis::RTrigBasis) 
   inds = zeros(Int, length(basis))
   inds[1] = 0 
   for k = 1:basis.N 
      inds[2*k] = k
      inds[2*k+1] = -k
   end
   return inds
end

_valtype(basis::RTrigBasis, T::Type{<: Real}) = T
_valtype(::RTrigBasis, T::Type{<: Hyper{<: Real}}) = T


function evaluate!(P::AbstractVector, basis::RTrigBasis, θ::Real)
   N = basis.N 
   @assert N  >= 1 
   @assert length(P) >= length(basis) # 2N+1

   P[1] = 1
   for k = 1:N 
      sk, ck = sincos(k * θ)
      @inbounds P[2*k] = ck
      @inbounds P[2*k+1] = sk
   end
   return P 
end 



function evaluate!(P::AbstractMatrix, basis::RTrigBasis, 
                   θ::AbstractVector)
   N = basis.N 
   nX = length(θ)
   @assert N  >= 1 
   @assert size(P, 2) >= length(basis) # 2N+1
   @assert size(P, 1) >= nX

   @inbounds begin 
      @simd ivdep for i = 1:nX 
         P[i, 1] = 1
      end

      for k = 1:N 
         @simd ivdep for i = 1:nX 
            sk, ck = sincos(k * θ[i])
            P[i, 2*k] = ck
            P[i, 2*k+1] = sk
         end
      end
   end
   return P 
end 

function evaluate_ed!(P::AbstractVector, dP::AbstractVector, 
                      basis::RTrigBasis, θ::Real)
   N = basis.N 
   nX = length(θ)
   @assert N  >= 1 
   @assert length(P) >= length(basis)  
   @assert length(dP) >= length(basis) 

   @inbounds begin 
      P[1] = 1
      dP[1] = 0
      for k = 1:N 
         sk, ck = sincos(k * θ)
         P[2*k] = ck
         P[2*k+1] = sk
         dP[2*k] = -k*sk
         dP[2*k+1] = k*ck
      end
   end
   return P, dP 
end 


function evaluate_ed!(P::AbstractMatrix, dP::AbstractMatrix, basis::RTrigBasis, 
                      θ::AbstractVector)
   N = basis.N 
   nX = length(θ)
   @assert N  >= 1 
   @assert size(P, 2) >= length(basis) # 2N+1
   @assert size(P, 1) >= nX
   @assert size(dP, 2) >= length(basis) # 2N+1
   @assert size(dP, 1) >= nX

   @inbounds begin 
      @simd ivdep for i = 1:nX 
         P[i, 1] = 1
         dP[i, 1] = 0
      end

      for k = 1:N 
         @simd ivdep for i = 1:nX 
            sk, ck = sincos(k * θ[i])

            P[i, 2*k] = ck
            P[i, 2*k+1] = sk
            dP[i, 2*k] = -k*sk
            dP[i, 2*k+1] = k*ck
         end
      end
   end
   return P, dP 
end 


function evaluate_ed2!(P::AbstractVector, dP::AbstractVector, ddP::AbstractVector,
                       basis::RTrigBasis, θ::Real)
   N = basis.N 
   @assert N  >= 1 
   @assert length(P) >= length(basis) # 2N+1
   @assert length(dP) >= length(basis) # 2N+1
   @assert length(ddP) >= length(basis) # 2N+1

   @inbounds begin 
      P[1] = 1
      dP[1] = 0
      ddP[1] = 0

      for k = 1:N 
         sk, ck = sincos(k * θ)
         P[2*k] = ck
         P[2*k+1] = sk
         dP[2*k] = -k*sk
         dP[2*k+1] = k*ck
         ddP[2*k] = -k^2*ck
         ddP[2*k+1] = -k^2*sk
      end
   end
   return P, dP, ddP 
end 



function evaluate_ed2!(P::AbstractMatrix, dP::AbstractMatrix, ddP::AbstractMatrix, basis::RTrigBasis, 
                      θ::AbstractVector)
   N = basis.N 
   nX = length(θ)
   @assert N  >= 1 
   @assert size(P, 2) >= length(basis) # 2N+1
   @assert size(P, 1) >= nX
   @assert size(dP, 2) >= length(basis) # 2N+1
   @assert size(dP, 1) >= nX
   @assert size(ddP, 2) >= length(basis) # 2N+1
   @assert size(ddP, 1) >= nX

   @inbounds begin 
      @simd ivdep for i = 1:nX 
         P[i, 1] = 1
         dP[i, 1] = 0
         ddP[i, 1] = 0
      end

      for k = 1:N 
         @simd ivdep for i = 1:nX 
            sk, ck = sincos(k * θ[i])

            P[i, 2*k] = ck
            P[i, 2*k+1] = sk
            dP[i, 2*k] = -k*sk
            dP[i, 2*k+1] = k*ck
            ddP[i, 2*k] = -k^2*ck
            ddP[i, 2*k+1] = -k^2*sk
         end
      end
   end
   return P, dP, ddP 
end 

