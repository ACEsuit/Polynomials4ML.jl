export BernsteinBasis

struct BernsteinBasis{N} <: AbstractP4MLBasis where {N} end

BernsteinBasis(N::Integer) = BernsteinBasis{N}()

Base.length(basis::BernsteinBasis{N}) where {N} = N

natural_indices(basis::BernsteinBasis) = 0:length(basis)-1

_valtype(basis::BernsteinBasis, T::Type{<:Real}) = T

_generate_input(basis::BernsteinBasis) = rand()  


using StaticArrays

function binom_svec(n::Int)
    return SVector{n+1}(binomial(n, k) for k in 0:n)
end

function _evaluate!(P, dP,
   basis::BernsteinBasis{N},
   x::AbstractVector{<:Real}) where {N}

   n = N - 1                     
   WITHGRAD = !isnothing(dP)
   @assert size(P) == (length(x), N)

   BNs = binom_svec(n)

   @simd ivdep for k = 0:n
      # binom = binomial(n, k)
      binom = BNs[k+1]
      for i in eachindex(x)
         xi = x[i]
         ominx = 1 - xi
         val = binom * xi^k * ominx^(n - k)
         P[i, k+1] = val
         a = binom * xi^(k-1) * ominx^(n - k - 1)
         ######
         ### There might be a more stable way to compute this 
         ######
         if WITHGRAD
            dP[i, k+1] = k * a * (1-xi) - (n - k)* a * xi
         end
      end
   end
   return nothing
end


@kernel function _ka_evaluate!(P, dP, basis::BernsteinBasis{N}, x::AbstractVector{T}
   ) where {T, N}
   i = @index(Global)
   @uniform WITHGRAD = !isnothing(dP)
   n = N - 1    

   BNs = binom_svec(n)

   @inbounds begin
      for k = 0:n
         # binom = binomial(n, k)
         binom = BNs[k+1]
         xi = x[i]
         ominx = 1 - xi
         val = binom * xi^k * ominx^(n - k)
         P[i, k+1] = val
         a = binom * xi^(k-1) * ominx^(n - k - 1)
         ######
         ### There might be a more stable way to compute this 
         ######
         if WITHGRAD
            dP[i, k+1] = k * a * (1-xi) - (n - k)* a * xi
         end
      end
   end
end

