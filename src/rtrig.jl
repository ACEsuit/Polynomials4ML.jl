
export RTrigBasis

"""
`RTrigBasis(N)`: 

Real trigonometric polynomials up to degree `N` (inclusive). The basis is ordered as 
```
[1, cos(θ), sin(θ), cos(2θ), sin(2θ), ..., cos(Nθ), sin(Nθ) ]
```
where `θ` is input variable. 
"""
struct RTrigBasis{N} <: AbstractP4MLBasis
end

RTrigBasis(N::Integer) = RTrigBasis{N}()

Base.length(basis::RTrigBasis{N}) where {N} = 2 * N + 1

function natural_indices_trig(N)
   inds = zeros(Int, 2*N+1)
   inds[1] = 0 
   for k = 1:N
      inds[2*k] = k
      inds[2*k+1] = -k
   end
   return [ (n = k,) for k in inds ]
end

natural_indices(basis::RTrigBasis{N}) where {N} = natural_indices_trig(N)

_valtype(basis::RTrigBasis, T::Type{<: Real}) = T

# _valtype(::RTrigBasis, T::Type{<: Hyper{<: Real}}) = T

_generate_input(basis::RTrigBasis) = 2 * π * rand() - π


##


function _evaluate!(P, dP, basis::RTrigBasis{N}, θ::BATCH, ps, st) where {N}
   nX = length(θ)
   @assert N  >= 1 
   WITHGRAD = !isnothing(dP)

   @inbounds begin 
      @simd ivdep for i = 1:nX 
         P[i, 1] = 1
         WITHGRAD && (dP[i, 1] = 0)
      end

      for k = 1:N 
         @simd ivdep for i = 1:nX 
            sk, ck = sincos(k * θ[i])
            P[i, 2*k] = ck
            P[i, 2*k+1] = sk
            WITHGRAD && (dP[i, 2*k] = -k*sk)
            WITHGRAD && (dP[i, 2*k+1] = k*ck)
         end
      end
   end
   return nothing 
end 




@kernel function _ka_evaluate!(P, dP, basis::RTrigBasis{N}, θ::BATCH) where {N}
   i = @index(Global)
   @uniform WITHGRAD = !isnothing(dP)
   @assert N  >= 1 

   @inbounds begin 
      P[i, 1] = 1
      WITHGRAD && (dP[i, 1] = 0)

      for k = 1:N 
         sk, ck = sincos(k * θ[i])
         P[i, 2*k] = ck
         P[i, 2*k+1] = sk
         WITHGRAD && (dP[i, 2*k] = -k*sk)
         WITHGRAD && (dP[i, 2*k+1] = k*ck)
      end
   end
   nothing 
end 