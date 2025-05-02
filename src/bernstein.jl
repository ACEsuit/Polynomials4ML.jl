export BernsteinBasis

struct BernsteinBasis{N} <: AbstractP4MLBasis where {N} end

BernsteinBasis(N::Integer) = BernsteinBasis{N}()

Base.length(basis::BernsteinBasis{N}) where {N} = N

natural_indices(basis::BernsteinBasis) = [ (n = n,) for n = 0:length(basis)-1 ]

_valtype(basis::BernsteinBasis, T::Type{<:Real}) = T

_generate_input(basis::BernsteinBasis) = rand()

@generated function static_binomial(::Val{N}) where {N}
   b = [ binomial(N - 1, k) for k in 0:N - 1 ]
   return :(($(b...),)) ## copilot suggested this :)
end


function _evaluate!(P, dP, basis::BernsteinBasis{N}, x::AbstractVector{<:Real}) where {N}

   n = N - 1                     
   WITHGRAD = !isnothing(dP)
   @assert size(P) == (length(x), N)

   BNs = static_binomial(Val(N))

   # k = 0  
   k = 0
   binom = BNs[k+1]
   for i in eachindex(x)
      xi = x[i]
      ominx = 1 - xi
      P[i, k+1] = binom * ominx^n
      if WITHGRAD
         dP[i, k+1] = -n * binom * ominx^(n - 1)
      end
   end

   # k = n 
   k = n
   binom = BNs[k+1]
   for i in eachindex(x)
      xi = x[i]
      P[i, k+1] = binom * xi^n
      if WITHGRAD
         dP[i, k+1] = n * binom * xi^(n - 1)
      end
   end

   # main loop 
   for k = 1:n-1
      binom = BNs[k+1]
      @simd ivdep for i in eachindex(x)
         xi = x[i]
         ominx = 1 - xi
         a = binom * xi^(k - 1) * ominx^(n - k - 1)
         P[i, k+1] = a * xi * ominx
         if WITHGRAD
               dP[i, k+1] = a * (k * ominx - (n - k) * xi)
         end
      end
   end
end


@kernel function _ka_evaluate!(P, dP, basis::BernsteinBasis{N}, x::AbstractVector{T}
   ) where {T, N}

   i = @index(Global)
   @uniform WITHGRAD = !isnothing(dP)
   n = N - 1    

   BNs = static_binomial(Val(N))

   @inbounds begin
      xi = x[i]
      ominx = 1 - xi

      # k = 0
      k = 0
      binom = BNs[k+1]
      P[i, k+1] = binom * ominx^n
      if WITHGRAD
         dP[i, k+1] = -n * binom * ominx^(n - 1)
      end

      # k = n
      k = n
      binom = BNs[k+1]
      P[i, k+1] = binom * xi^n
      if WITHGRAD
         dP[i, k+1] = n * binom * xi^(n - 1)
      end

      # main loop
      for k = 1:n-1
         binom = BNs[k+1]
         a = binom * xi^(k - 1) * ominx^(n - k - 1)
         P[i, k+1] = a * xi * ominx
         if WITHGRAD
            dP[i, k+1] = a * (k * ominx - (n - k) * xi)
         end
      end
   end
end

