export CTrigBasis


"""
Complex trigonometric polynomials up to degree `N` (inclusive). The basis is 
constructed in the order 
```
[1, exp(im*θ), exp(-im*θ), exp(2im*θ), exp(-2im*θ), ..., 
                                exp(N*im*θ), exp(-N*im*θ) ]
```
where `θ` is input variable. 
"""
struct CTrigBasis{N} <: AbstractP4MLBasis
end

CTrigBasis(N::Integer) = CTrigBasis{N}()

# natural_indices_trig is implemented in rtrig.jl 
natural_indices(basis::CTrigBasis{N}) where {N} = natural_indices_trig(N)

Base.length(basis::CTrigBasis{N}) where {N} = 2 * N + 1 

_valtype(basis::CTrigBasis, T::Type{<: Real}) = complex(T)

# _valtype(::CTrigBasis, T::Type{<: Hyper{<: Real}}) = complex(T)
  
_generate_input(basis::CTrigBasis) = 2 * π * rand() - π

# ----------------- main evaluation code 


function _evaluate!(P, dP, basis::CTrigBasis{N}, X::BATCH, ps, st) where {N}
   nX = length(X) 
   WITHGRAD = !isnothing(dP)
   @assert N >= 1

   @inbounds begin 
      for i = 1:nX 
         s, c = sincos(X[i])
         P[i, 1] = 1 
         P[i, 2] = Complex(c, s)
         P[i, 3] = Complex(c, -s)
         if WITHGRAD 
            dP[i, 1] = 0
            dP[i, 2] = Complex(-s, c)
            dP[i, 3] = Complex(-s, -c)
         end
      end 
      for n = 2:N 
         for i = 1:nX 
            P[i, 2n] = P[i, 2] * P[i, 2n-2]
            P[i, 2n+1] = P[i, 3] * P[i, 2n-1]
            if WITHGRAD 
               dP[i, 2n] = im * n * P[i, 2n]
               dP[i, 2n+1] = - im * n * P[i, 2n+1] 
            end
         end
      end
   end
   return nothing 
end



@kernel function _ka_evaluate!(P, dP, basis::CTrigBasis{N}, X::BATCH) where {N}
   i = @index(Global)
   WITHGRAD = !isnothing(dP)
   @assert N >= 1

   @inbounds begin 
      s, c = sincos(X[i])
      P[i, 1] = 1 
      P[i, 2] = Complex(c, s)
      P[i, 3] = Complex(c, -s)
      if WITHGRAD 
         dP[i, 1] = 0
         dP[i, 2] = Complex(-s, c)
         dP[i, 3] = Complex(-s, -c)
      end
      for n = 2:N 
         P[i, 2n] = P[i, 2] * P[i, 2n-2]
         P[i, 2n+1] = P[i, 3] * P[i, 2n-1]
         if WITHGRAD 
            dP[i, 2n] = im * n * P[i, 2n]
            dP[i, 2n+1] = - im * n * P[i, 2n+1] 
         end
      end
   end
   nothing 
end



