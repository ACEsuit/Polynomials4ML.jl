export RTrigPolys, CTrigBasis


"""
Complex trigonometric polynomials up to degree `N` (inclusive). The basis is 
constructed in the order 
```
[1, exp(im*θ), exp(-im*θ), exp(2im*θ), exp(-2im*θ), ..., 
                                exp(N*im*θ), exp(-N*im*θ) ]
```
where `θ` is input variable. These polynomials are orthonormal w.r.t. the 
normalized L2-inner product on the torus. 
"""
struct CTrigBasis{T} <: PolyBasis4ML
   N::Int
   # ----------------- metadata 
   meta::Dict{String, Any}
end

CTrigBasis(N::Integer, T = Float64, meta = Dict{String, Any}()) = 
         CTrigBasis{T}(N, meta)

# TODO 
# struct RTrigPolys{T}
#    N::Int
#    # ----------------- metadata 
#    meta::Dict{String, Any}
# end


# ----------------- interface functions 



natural_indices(basis::CTrigBasis) = -basis.N:basis.N 

index(basis::CTrigBasis, m::Integer) = 
         2 * abs(m) + (sign(m) <= 0 ? 1 : 0)


Base.length(basis::CTrigBasis) = 2 * basis.N + 1 

_alloc(basis::CTrigBasis{T1}, x::T2) where {T1, T2 <: Number} = 
            zeros(promote_type(Complex{T1}, T2), length(basis))

_alloc(basis::CTrigBasis{T1}, X::AbstractVector{T2}) where {T1, T2 <: Number} = 
            zeros(promote_type(Complex{T1}, T2), length(X), length(basis))


            
# ----------------- main evaluation code 

function evaluate!(P, basis::CTrigBasis, x) 
   N = basis.N 
   @assert length(P) >= length(basis) # 2N+1
   @inbounds P[1] = 1 
   @inbounds if N > 0 
      s, c = sincos(x)
      z = Complex(c, s)
      zi = Complex(c, -s)
      P[2] = z
      P[3] = zi
      for n = 2:N 
         P[2*n] = z * P[2*n-2]
         P[2*n+1] = zi * P[2*n-1]
      end
   end
   return P 
end

function evaluate_ed!(P, dP, basis::CTrigBasis, x) 
   N = basis.N 
   @assert length(P) >= length(basis) # 2N+1
   @assert length(dP) >= length(basis) 
   @inbounds P[1] = 1 
   @inbounds dP[1] = 0 
   @inbounds if N > 0 
      s, c = sincos(x)
      z = Complex(c, s)
      zi = Complex(c, -s)
      P[2] = z
      dP[2] = im * z 
      P[3] = zi
      dP[3] = -im * zi 
      for n = 2:N 
         P[2*n] = z * P[2*n-2]
         dP[2*n] = im * n * P[2*n]
         P[2*n+1] = zi * P[2*n-1]
         dP[2*n+1] = -im * n * P[2*n+1]
      end
   end
   return P, dP 
end

function evaluate_ed2!(P, dP, ddP, basis::CTrigBasis, x) 
   N = basis.N 
   @assert length(P) >= length(basis) # 2N+1
   @assert length(dP) >= length(basis) 
   @assert length(ddP) >= length(basis) 
   @inbounds P[1] = 1 
   @inbounds dP[1] = 0 
   @inbounds ddP[1] = 0 
   @inbounds if N > 0 
      s, c = sincos(x)
      z = Complex(c, s)
      zi = Complex(c, -s)
      P[2] = z
      dP[2] = im * z 
      ddP[2] = - z 
      P[3] = zi
      dP[3] = -im * zi 
      ddP[3] = - zi 
      for n = 2:N 
         P[2*n] = z * P[2*n-2]
         dP[2*n] = im * n * P[2*n]
         ddP[2*n] = - n^2 * P[2*n]
         P[2*n+1] = zi * P[2*n-1]
         dP[2*n+1] = -im * n * P[2*n+1]
         ddP[2*n+1] = - n^2 * P[2*n+1]
      end
   end
   return P, dP, ddP 
end



function evaluate!(P, basis::CTrigBasis, X::AbstractVector)
   N = basis.N 
   nX = length(X) 
   @assert size(P, 1) >= length(X) 
   @assert size(P, 2) >= length(basis)  # 2N+1

   @inbounds begin 
      for i = 1:nX 
         P[i, 1] = 1 
         s, c = sincos(X[i])
         P[i, 2] = Complex(c, s)
         P[i, 3] = Complex(c, -s)
      end 
      for n = 2:N 
         for i = 1:nX 
            P[i, 2n] = P[i, 2] * P[i, 2n-2]
            P[i, 2n+1] = P[i, 3] * P[i, 2n-1]
         end
      end
   end
   return P 
end



function evaluate_ed!(P, dP, basis::CTrigBasis, X::AbstractVector)
   N = basis.N 
   nX = length(X) 
   @assert size(P, 1) >= length(X) 
   @assert size(P, 2) >= length(basis)  # 2N+1
   @assert size(dP, 1) >= length(X) 
   @assert size(dP, 2) >= length(basis)

   @inbounds begin 
      for i = 1:nX 
         P[i, 1] = 1 
         s, c = sincos(X[i])
         P[i, 2] = Complex(c, s)
         dP[i, 2] = Complex(-s, c)
         P[i, 3] = Complex(c, -s)
         dP[i, 3] = Complex(-s, -c)
      end 
      for n = 2:N 
         for i = 1:nX 
            P[i, 2n] = P[i, 2] * P[i, 2n-2]
            dP[i, 2n] = im * n * P[i, 2n]
            P[i, 2n+1] = P[i, 3] * P[i, 2n-1]
            dP[i, 2n+1] = - im * n * P[i, 2n+1] 
         end
      end
   end
   return P, dP 
end



function evaluate_ed2!(P, dP, ddP, basis::CTrigBasis, X::AbstractVector)
   N = basis.N 
   nX = length(X) 
   @assert size(P, 1) >= length(X) 
   @assert size(P, 2) >= length(basis)  # 2N+1
   @assert size(dP, 1) >= length(X) 
   @assert size(dP, 2) >= length(basis)
   @assert size(ddP, 1) >= length(X) 
   @assert size(ddP, 2) >= length(basis)

   @inbounds begin 
      for i = 1:nX 
         P[i, 1] = 1 
         s, c = sincos(X[i])
         P[i, 2] = Complex(c, s)
         dP[i, 2] = Complex(-s, c)
         ddP[i, 2] = Complex(-c, -s)
         P[i, 3] = Complex(c, -s)
         dP[i, 3] = Complex(-s, -c)
         ddP[i, 3] = Complex(-c, s)
      end 
      for n = 2:N 
         for i = 1:nX 
            P[i, 2n] = P[i, 2] * P[i, 2n-2]
            dP[i, 2n] = im * n * P[i, 2n]
            ddP[i, 2n] = im * n * dP[i, 2n]
            P[i, 2n+1] = P[i, 3] * P[i, 2n-1]
            dP[i, 2n+1] = - im * n * P[i, 2n+1] 
            ddP[i, 2n+1] = - im * n * dP[i, 2n+1] 
         end
      end
   end
   return P, dP, ddP 
end

