module RT 

import Polynomials4ML
import Polynomials4ML: evaluate, evaluate!, 
                       evaluate_d, evaluate_d!, 
                       evaluate_dd, evaluate_dd!, 
                       natural_indices

struct RTrigBasis
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

(basis::RTrigBasis)(θ) = evaluate(basis, θ)

function evaluate(basis::RTrigBasis, θ::T) where {T <: Real}
   P = zeros(T, basis.N*2+1)
   evaluate!(P, basis, θ)
   return P 
end

function evaluate(basis::RTrigBasis, θ::AbstractVector{T}) where {T <: Real}
   P = zeros(T, length(θ), basis.N*2+1)
   evaluate!(P, basis, θ)
   return P 
end


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
   return nothing 
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
   return nothing 
end 


end
