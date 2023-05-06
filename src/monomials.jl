

export MonoBasis


"""
Standard Monomials basis. This should very rarely be used. Possibly useful in combination with a transformation of the inputs, e.g. exponential.
"""
struct MonoBasis <: AbstractPoly4MLBasis
   N::Int
   # ----------------- metadata 
   meta::Dict{String, Any}
end

MonoBasis(N::Integer, meta = Dict{String, Any}()) = 
         MonoBasis(N, meta)


# ----------------- interface functions 



natural_indices(basis::MonoBasis) = 0:basis.N

index(basis::MonoBasis, m::Integer) = m+1

Base.length(basis::MonoBasis) = basis.N+1

_valtype(basis::MonoBasis, x::Number) = typeof(x)

_alloc(basis::MonoBasis, x::T2) where {T2 <: Number} = 
            zeros(T2, length(basis))

_alloc(basis::MonoBasis, X::AbstractVector{T2}) where {T2 <: Number} = 
            zeros(T2, length(X), length(basis))

            
# ----------------- main evaluation code 

function evaluate!(P, basis::MonoBasis, x::Number) 
   N = basis.N 
   @assert length(P) >= length(basis) 
   @inbounds P[1] = 1 
   @inbounds for n = 1:N 
      P[n+1] = x * P[n]
   end
   return P 
end

function evaluate!(P, basis::MonoBasis, X::AbstractVector)
   N = basis.N 
   nX = length(X) 
   @assert size(P, 2) >= N+1
   @assert size(P, 1) >= nX 
   @inbounds begin 
      @simd ivdep for i = 1:nX 
         P[i, 1] = 1 
      end
      for n = 1:N 
         @simd ivdep for i = 1:nX
            P[i, n+1] = X[i] * P[i, n]
         end
      end
   end
   return P 
end

function evaluate_ed!(P, dP, basis::MonoBasis, x) 
   N = basis.N 
   @assert length(P) >= length(basis) 
   @assert length(dP) >= length(basis) 
   @inbounds P[1] = 1 
   @inbounds dP[1] = 0 
   @inbounds for n = 1:N 
      P[n+1] = x * P[n] 
      dP[n+1] = n * P[n]
   end
   return P, dP 
end

function evaluate_ed2!(P, dP, ddP, basis::MonoBasis, x) 
   N = basis.N 
   @assert length(P) >= length(basis) 
   @assert length(dP) >= length(basis) 
   @assert length(ddP) >= length(basis) 
   @inbounds P[1] = 1 
   @inbounds dP[1] = 0 
   @inbounds ddP[1] = 0 

   @inbounds if N > 0 
      P[n+1] = x
      dP[n+1] = 1
      ddP[2] = 0 
      for n = 2:N 
         P[n+1] = x * P[n]
         dP[n+1] = n * P[n]
         ddP[n+1] = n*(n-1) * P[n-1]
      end
   end
   return P, dP, ddP 
end



# function evaluate!(P, basis::MonoBasis, X::AbstractVector)
#    N = basis.N 
#    nX = length(X) 
#    @assert size(P, 1) >= length(X) 
#    @assert size(P, 2) >= length(basis)  # 2N+1

#    @inbounds begin 
#       for i = 1:nX 
#          P[i, 1] = 1 
#          s, c = sincos(X[i])
#          P[i, 2] = Complex(c, s)
#          P[i, 3] = Complex(c, -s)
#       end 
#       for n = 2:N 
#          for i = 1:nX 
#             P[i, 2n] = P[i, 2] * P[i, 2n-2]
#             P[i, 2n+1] = P[i, 3] * P[i, 2n-1]
#          end
#       end
#    end
#    return P 
# end



# function evaluate_ed!(P, dP, basis::MonoBasis, X::AbstractVector)
#    N = basis.N 
#    nX = length(X) 
#    @assert size(P, 1) >= length(X) 
#    @assert size(P, 2) >= length(basis)  # 2N+1
#    @assert size(dP, 1) >= length(X) 
#    @assert size(dP, 2) >= length(basis)

#    @inbounds begin 
#       for i = 1:nX 
#          P[i, 1] = 1 
#          s, c = sincos(X[i])
#          P[i, 2] = Complex(c, s)
#          dP[i, 2] = Complex(-s, c)
#          P[i, 3] = Complex(c, -s)
#          dP[i, 3] = Complex(-s, -c)
#       end 
#       for n = 2:N 
#          for i = 1:nX 
#             P[i, 2n] = P[i, 2] * P[i, 2n-2]
#             dP[i, 2n] = im * n * P[i, 2n]
#             P[i, 2n+1] = P[i, 3] * P[i, 2n-1]
#             dP[i, 2n+1] = - im * n * P[i, 2n+1] 
#          end
#       end
#    end
#    return P, dP 
# end



# function evaluate_ed2!(P, dP, ddP, basis::MonoBasis, X::AbstractVector)
#    N = basis.N 
#    nX = length(X) 
#    @assert size(P, 1) >= length(X) 
#    @assert size(P, 2) >= length(basis)  # 2N+1
#    @assert size(dP, 1) >= length(X) 
#    @assert size(dP, 2) >= length(basis)
#    @assert size(ddP, 1) >= length(X) 
#    @assert size(ddP, 2) >= length(basis)

#    @inbounds begin 
#       for i = 1:nX 
#          P[i, 1] = 1 
#          s, c = sincos(X[i])
#          P[i, 2] = Complex(c, s)
#          dP[i, 2] = Complex(-s, c)
#          ddP[i, 2] = Complex(-c, -s)
#          P[i, 3] = Complex(c, -s)
#          dP[i, 3] = Complex(-s, -c)
#          ddP[i, 3] = Complex(-c, s)
#       end 
#       for n = 2:N 
#          for i = 1:nX 
#             P[i, 2n] = P[i, 2] * P[i, 2n-2]
#             dP[i, 2n] = im * n * P[i, 2n]
#             ddP[i, 2n] = im * n * dP[i, 2n]
#             P[i, 2n+1] = P[i, 3] * P[i, 2n-1]
#             dP[i, 2n+1] = - im * n * P[i, 2n+1] 
#             ddP[i, 2n+1] = - im * n * dP[i, 2n+1] 
#          end
#       end
#    end
#    return P, dP, ddP 
# end

