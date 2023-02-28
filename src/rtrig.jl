module RT 

import Polynomials4ML
using Polynomials4ML: PolyBasis4ML
import Polynomials4ML: evaluate!, 
                       evaluate_ed!, 
                       evaluate_ed2!, 
                       natural_indices, _alloc

struct RTrigBasis <: PolyBasis4ML
   N::Int
   # ----------------- metadata 
   meta::Dict{String, Any}
end

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

RTrigBasis(N::Integer, meta = Dict{String, Any}()) = 
         RTrigBasis{T}(N, meta)


_alloc(basis::RTrigBasis, x::Real) = zeros(typeof(x), length(basis))

_alloc(basis::RTrigBasis, x::AbstractVector{<: Real}) = zeros(eltype(x), length(x), length(basis))


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
                   θ::AbstractVector{<: Real})
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

function evaluate_ed!(P::AbstractMatrix, dP::AbstractMatrix, basis::RTrigBasis, 
                      θ::AbstractVector{<: Real})
   N = basis.N 
   nX = length(θ)
   @assert N  >= 1 
   @assert size(P, 2) >= length(basis) # 2N+1
   @assert size(P, 1) >= nX

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


function evaluate_ed2!(P::AbstractMatrix, dP::AbstractMatrix, ddP::AbstractMatrix, basis::RTrigBasis, 
                      θ::AbstractVector{<: Real})
   N = basis.N 
   nX = length(θ)
   @assert N  >= 1 
   @assert size(P, 2) >= length(basis) # 2N+1
   @assert size(P, 1) >= nX

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



end
