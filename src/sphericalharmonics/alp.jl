"""
`ALPolynomials` : an auxiliary datastructure for
evaluating the associated lagrange functions
used for the spherical harmonics
Constructor:
```julia
ALPolynomials(maxL::Integer, T::Type=Float64)
```
"""
struct ALPolynomials{T} 
	L::Int
	A::Vector{T}
	B::Vector{T}
	pool::ArrayCache{T, 1}
	ppool::ArrayCache{T, 2}
   pool_t::ArrayCache{T, 1}
end


ALPolynomials(L::Integer, A::Vector{T}, B::Vector{T}) where {T}  = 
		ALPolynomials(L, A, B, 
                    ArrayCache{T, 1}(), 
                    ArrayCache{T, 2}(), 
                    ArrayCache{T, 1}() )

Base.length(alp::ALPolynomials) = sizeP(alp.L)

import Base.==
==(B1::ALPolynomials{T}, B2::ALPolynomials{T}) where {T} = 
		((B1.L == B2.L) && (B1.A ≈ B2.A) && (B1.B ≈ B2.B))


_valtype(alp::ALPolynomials{T}, x::SphericalCoords{S}) where {T, S} = 
			promote_type(T, S) 


# ---------------------- Indexing

"""
`sizeP(maxL):` 
Return the size of the set of Associated Legendre Polynomials ``P_l^m(x)`` of
degree less than or equal to the given maximum degree
"""
sizeP(maxL) = div((maxL + 1) * (maxL + 2), 2)

"""
`index_p(l,m):`
Return the index into a flat array of Associated Legendre Polynomials `P_l^m`
for the given indices `(l,m)`. `P_l^m` are stored in l-major order i.e. 
```
	[P(0,0), [P(1,0), P(1,1), P(2,0), ...]
```
"""
index_p(l::Integer,m::Integer) = m + div(l*(l+1), 2) + 1


# -------------------- construct the recurrance relation 


function ALPolynomials(L::Integer, T::Type=Float64)
	# Precompute coefficients ``a_l^m`` and ``b_l^m`` for all l <= L, m <= l
	alp = ALPolynomials(L, zeros(T, sizeP(L)), zeros(T, sizeP(L)))
	for l in 2:L
		ls = l*l
		lm1s = (l-1) * (l-1)
		for m in 0:(l-2)
			ms = m * m
			alp.A[index_p(l, m)] = sqrt((4 * ls - 1.0) / (ls - ms))
			alp.B[index_p(l, m)] = -sqrt((lm1s - ms) / (4 * lm1s - 1.0))
		end
	end
	return alp
end



# -------------------- evaluation interface

function evaluate(alp::ALPolynomials, S::SphericalCoords) 
	P = acquire!(alp.pool, length(alp), _valtype(alp, S))
	evaluate!(parent(P), alp, S)
	return P 
end

function evaluate(alp::ALPolynomials, S::AbstractVector{<: SphericalCoords}) 
	P = acquire!(alp.ppool, (length(S), length(alp)), _valtype(alp, S[1]))
	evaluate!(parent(P), alp, S)
	return P 
end

function _evaluate_ed(alp::ALPolynomials, S::SphericalCoords) 
	VT = _valtype(alp, S)
	P = acquire!(alp.pool, length(alp), VT)
	dP = acquire!(alp.pool, length(alp), VT)
	_evaluate_ed!(parent(P), parent(dP), alp::ALPolynomials, S::SphericalCoords)
	return P, dP 
end


# -------------------- serial evaluation codes


function evaluate!(P, alp::ALPolynomials, S::SphericalCoords)
	L = alp.L 
	A = alp.A 
	B = alp.B 
	@assert length(A) >= sizeP(L)
	@assert length(B) >= sizeP(L)
	@assert length(P) >= sizeP(L)

	temp = sqrt(0.5/π)
	P[index_p(0, 0)] = temp
	if L == 0; return P; end

	P[index_p(1, 0)] = S.cosθ * sqrt(3) * temp
	temp = - sqrt(1.5) * S.sinθ * temp
	P[index_p(1, 1)] = temp

	for l in 2:L
		il = ((l*(l+1)) ÷ 2) + 1
		ilm1 = il - l
		ilm2 = ilm1 - l + 1
		for m in 0:(l-2)
			@inbounds P[il+m] = A[il+m] * (     S.cosθ * P[ilm1+m]
  					                           + B[il+m] * P[ilm2+m] )
		end
		@inbounds P[il+l-1] = S.cosθ * sqrt(2 * (l - 1) + 3) * temp
		temp = -sqrt(1.0 + 0.5 / l) * S.sinθ * temp
		@inbounds P[il+l] = temp
	end

	return P
end



# this doesn't use the standard name because it doesn't 
# technically perform the derivative w.r.t. S, but w.r.t. θ
# further, P doesn't store P but (P if m = 0) or (P * sinθ if m > 0)
# this is done for numerical stability 
function _evaluate_ed!(P, dP, alp::ALPolynomials, S::SphericalCoords)
	L = alp.L 
	A = alp.A 
	B = alp.B 
	@assert length(A) >= sizeP(L)
	@assert length(B) >= sizeP(L)
	@assert length(P) >= sizeP(L)
	@assert length(dP) >= sizeP(L)

	temp = sqrt(0.5/π)
	P[index_p(0, 0)] = temp
	temp_d = 0.0
	dP[index_p(0, 0)] = temp_d
	if L == 0; return P, dP; end

	P[index_p(1, 0)] = S.cosθ * sqrt(3) * temp
	dP[index_p(1, 0)] = -S.sinθ * sqrt(3) * temp + S.cosθ * sqrt(3) * temp_d
	temp1, temp_d = ( - sqrt(1.5) * temp,
					      - sqrt(1.5) * (S.cosθ * temp + S.sinθ * temp_d) )
	P[index_p(1, 1)] = temp1
	dP[index_p(1, 1)] = temp_d

	for l in 2:L
		m = 0
		@inbounds P[index_p(l, m)] =
				A[index_p(l, m)] * (     S.cosθ * P[index_p(l - 1, m)]
				             + B[index_p(l, m)] * P[index_p(l - 2, m)] )
		@inbounds dP[index_p(l, m)] =
			A[index_p(l, m)] * (
							- S.sinθ * P[index_p(l - 1, m)]
							+ S.cosθ * dP[index_p(l - 1, m)]
			             + B[index_p(l, m)] * dP[index_p(l - 2, m)] )

		for m in 1:(l-2)
			@inbounds P[index_p(l, m)] =
					A[index_p(l, m)] * (     S.cosθ * P[index_p(l - 1, m)]
					             + B[index_p(l, m)] * P[index_p(l - 2, m)] )
			@inbounds dP[index_p(l, m)] =
				A[index_p(l, m)] * (
								- S.sinθ^2 * P[index_p(l - 1, m)]
								+ S.cosθ * dP[index_p(l - 1, m)]
				             + B[index_p(l, m)] * dP[index_p(l - 2, m)] )
		end
		@inbounds P[index_p(l, l - 1)] = sqrt(2 * (l - 1) + 3) * S.cosθ * temp1
		@inbounds dP[index_p(l, l - 1)] = sqrt(2 * (l - 1) + 3) * (
									        -S.sinθ^2 * temp1 + S.cosθ * temp_d )

      (temp1, temp_d) = (
					-sqrt(1.0+0.5/l) * S.sinθ * temp1,
		         -sqrt(1.0+0.5/l) * (S.cosθ * temp1 * S.sinθ + S.sinθ * temp_d) )
		@inbounds P[index_p(l, l)] = temp1
		@inbounds dP[index_p(l, l)] = temp_d
	end

	return P, dP
end



# ------------------ batched evaluation 


function evaluate!(P, alp::ALPolynomials, 
                   S::AbstractVector{<: SphericalCoords} )
	L = alp.L 
	A = alp.A 
	B = alp.B 
   nS = length(S) 
	@assert length(A) >= sizeP(L)
	@assert length(B) >= sizeP(L)
	@assert size(P, 2) >= sizeP(L)
	@assert size(P, 1) >= length(S)

   # NB: There are 3 minimal allocations here because a CachedArray is not 
   #     a bitstype. Removing this allocation appears to improve the performance 
   #     but around 1%, so not even close to worthwhile. 
   _t = acquire!(alp.pool_t, length(S)) # temp from the serial version 
   _co = acquire!(alp.pool_t, length(S)) 
   _si = acquire!(alp.pool_t, length(S)) 
   t = parent(_t)
   co = parent(_co)
   si = parent(_si)

   @inbounds begin 
      t0 = sqrt(0.5/π)
      i_p00 = index_p(0, 0)
      for i=1:nS
         P[i, i_p00] = t[i] = t0  
         co[i] = S[i].cosθ
         si[i] = S[i].sinθ
      end
      if L == 0; return P; end

      for i = 1:nS
         P[i, index_p(1, 0)] = co[i] * sqrt(3) * t[i]
         t[i] = - sqrt(1.5) * si[i] * t[i]
         P[i, index_p(1, 1)] = t[i]
      end

      for l in 2:L
         il = ((l*(l+1)) ÷ 2) + 1
         ilm1 = il - l
         ilm2 = ilm1 - l + 1
         for m in 0:(l-2)
            i0 = il+m 
            i1 = ilm1+m 
            i2 = ilm2+m 
            a = A[i0] 
            b = B[i0] 
            @simd for i = 1:nS
               P[i, i0] = a * fma(co[i], P[i, i1], b * P[i, i2])
            end
         end

         a = sqrt(2 * (l - 1) + 3)
         b = -sqrt(1.0 + 0.5 / l)
         i1 = il+l-1 
         i2 = il+l 
         @simd for i = 1:nS
            @fastmath P[i, i1] = a * co[i] * t[i]
            @fastmath t[i] = b * si[i] * t[i]
            P[i, i2] = t[i]
         end
      end

   end 

   release!(_co)
   release!(_si)
   release!(_t)
	return P
end
