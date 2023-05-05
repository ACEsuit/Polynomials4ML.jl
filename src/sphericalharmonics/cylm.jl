


"""
complex spherical harmonics
"""
struct CYlmBasis{T} <: AbstractPoly4MLBasis
	alp::ALPolynomials{T}
   # ----------------------------
	pool::ArrayCache{Complex{T}, 1}
   ppool::ArrayCache{Complex{T}, 2}
	pool_d::ArrayCache{SVector{3, Complex{T}}, 1}
   ppool_d::ArrayCache{SVector{3, Complex{T}}, 2}
	tmp_s::TempArray{SphericalCoords{T}, 1}
	tmp_t::TempArray{Complex{T}, 1}
	tmp_sin::TempArray{T, 1}
	tmp_cos::TempArray{T, 1}
end

CYlmBasis(maxL::Integer, T::Type=Float64) = 
      CYlmBasis(ALPolynomials(maxL, T))

CYlmBasis(alp::ALPolynomials{T}) where {T} = 
      CYlmBasis(alp, 
		         ArrayCache{Complex{T}, 1}(), 
               ArrayCache{Complex{T}, 2}(), 
               ArrayCache{SVector{3, Complex{T}}, 1}(), 
               ArrayCache{SVector{3, Complex{T}}, 2}(), 
					TempArray{SphericalCoords{T}, 1}(),
					TempArray{Complex{T}, 1}(), 
					TempArray{T, 1}(), 
					TempArray{T, 1}() )

Base.show(io::IO, basis::CYlmBasis) = 
      print(io, "CYlmBasis(L=$(maxL(basis)))")

"""
max L degree for which the alp coefficients have been precomputed
"""
maxL(sh::CYlmBasis) = sh.alp.L

_valtype(sh::CYlmBasis{T}, x::AbstractVector{S}) where {T <: Real, S <: Real} = 
			Complex{promote_type(T, S)}

_gradtype(sh::CYlmBasis{T}, x::AbstractVector{S})  where {T <: Real, S <: Real} = 
			SVector{3, Complex{promote_type(T, S)}}

import Base.==
==(B1::CYlmBasis, B2::CYlmBasis) =
		(B1.alp == B2.alp) && (typeof(B1) == typeof(B2))

Base.length(basis::CYlmBasis) = sizeY(maxL(basis))


# ---------------------- FIO

# write_dict(SH::CYlmBasis{T}) where {T} =
# 		Dict("__id__" => "ACE_CYlmBasis",
# 			  "T" => write_dict(T),
# 			  "maxL" => maxL(SH))

# read_dict(::Val{:ACE_CYlmBasis}, D::Dict) =
# 		CYlmBasis(D["maxL"], read_dict(D["T"]))



# ---------------------- Indexing

"""
`sizeY(maxL):`
Return the size of the set of spherical harmonics ``Y_{l,m}(θ,φ)`` of
degree less than or equal to the given maximum degree `maxL`
"""
sizeY(maxL) = (maxL + 1) * (maxL + 1)

"""
`index_y(l,m):`
Return the index into a flat array of real spherical harmonics `Y_lm`
for the given indices `(l,m)`. `Y_lm` are stored in l-major order i.e.
```
	[Y(0,0), Y(1,-1), Y(1,0), Y(1,1), Y(2,-2), ...]
```
"""
index_y(l::Integer, m::Integer) = m + l + (l*l) + 1

"""
Inverse of `index_y`: given an index into a vector of Ylm values, return the 
`l, m` indices.
"""
function idx2lm(i::Integer) 
	l = floor(Int, sqrt(i-1) + 1e-10)
	m = i - (l + (l*l) + 1)
	return l, m 
end 

idx2l(i::Integer) = floor(Int, sqrt(i-1) + 1e-10)


# ---------------------- evaluation interface code 

function evaluate(basis::CYlmBasis, x::AbstractVector{<: Real})
	Y = acquire!(basis.pool, length(basis), _valtype(basis, x))
	evaluate!(parent(Y), basis, x)
	return Y 
end

function evaluate!(Y, basis::CYlmBasis, x::AbstractVector{<: Real})
	L = maxL(basis)
	S = cart2spher(x) 
	P = evaluate(basis.alp, S)
	cYlm!(Y, maxL(basis), S, P)
	release!(P)
	return Y
end

function evaluate(basis::CYlmBasis, X::AbstractVector{<: AbstractVector})
	Y = acquire!(basis.ppool, (length(X), length(basis)))
	evaluate!(parent(Y), basis, X)
	return Y 
end

function evaluate!(Y, basis::CYlmBasis, 
						 X::AbstractVector{<: AbstractVector})
	L = maxL(basis)
	S = acquire!(basis.tmp_s, length(X))
	map!(cart2spher, S, X)
	P = evaluate(basis.alp, S)
	cYlm!(Y, maxL(basis), S, P, basis)
	release!(P)
	return Y
end


function evaluate_d(SH::CYlmBasis, 
						 R::Union{AbstractVector{<: Real}, 
						 			 AbstractVector{<: AbstractVector}} )
	B, dB = evaluate_ed(SH, R) 
	release!(B)
	return dB 
end 

function evaluate_ed(SH::CYlmBasis, R::AbstractVector{<: Real})
	Y = acquire!(SH.pool, length(SH), _valtype(SH, R))
	dY = acquire!(SH.pool_d, length(SH), _gradtype(SH, R))
	evaluate_ed!(parent(Y), parent(dY), SH, R)
	return Y, dY
end

function evaluate_ed!(Y, dY, SH::CYlmBasis, R::AbstractVector{<: Real})
	L = maxL(SH)
	S = cart2spher(R)
	P, dP = _evaluate_ed(SH.alp, S)
	cYlm_ed!(Y, dY, maxL(SH), S, P, dP)
	release!(P)
	release!(dP)
	return Y, dY
end


function evaluate_ed(SH::CYlmBasis, R::AbstractVector{<: AbstractVector})
	nR = length(R); nY = length(SH)
	Y = acquire!(SH.ppool, (nR, nY), _valtype(SH, R[1]))
	dY = acquire!(SH.ppool_d, (nR, nY), _gradtype(SH, R[1]))
	evaluate_ed!(parent(Y), parent(dY), SH, R)
	return Y, dY
end

function evaluate_ed!(Y, dY, SH::CYlmBasis, R::AbstractVector{<: AbstractVector})
	L = maxL(SH)
	S = acquire!(SH.tmp_s, length(R))
	map!(cart2spher, S, R)
	P, dP = _evaluate_ed(SH.alp, S)
	cYlm_ed!(Y, dY, maxL(SH), S, P, dP, SH)
	release!(P)
	release!(dP)
	return Y, dY
end

# ---------------------- serial evaluation code 



"""
evaluate complex spherical harmonics
"""
function cYlm!(Y, L, S::SphericalCoords, P::AbstractVector)
	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)
   @assert abs(S.cosθ) <= 1.0

	ep = 1 / sqrt(2) + im * 0
	for l = 0:L
		Y[index_y(l, 0)] = P[index_p(l, 0)] * ep
	end

   sig = 1
   ep_fact = S.cosφ + im * S.sinφ
	for m in 1:L
		sig *= -1
		ep *= ep_fact            # ep =   exp(i *   m  * φ)
		em = sig * conj(ep)      # em = ± exp(i * (-m) * φ)
		for l in m:L
			p = P[index_p(l,m)]
         # (-1)^m * p * exp(-im*m*phi) / sqrt(2)
			@inbounds Y[index_y(l, -m)] = em * p  
         #          p * exp( im*m*phi) / sqrt(2) 
			@inbounds Y[index_y(l,  m)] = ep * p   
		end
	end

	return Y
end



"""
evaluate gradients of complex spherical harmonics
"""
function cYlm_ed!(Y, dY, L, S::SphericalCoords, P, dP)
	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)
	@assert length(dY) >= sizeY(L)

	# m = 0 case
	ep = 1 / sqrt(2)
	for l = 0:L
		Y[index_y(l, 0)] = P[index_p(l, 0)] * ep
		dY[index_y(l, 0)] = dspher_to_dcart(S, 0.0, dP[index_p(l, 0)] * ep)
	end

   sig = 1
   ep_fact = S.cosφ + im * S.sinφ

	for m in 1:L
		sig *= -1
		ep *= ep_fact            # ep =   exp(i *   m  * φ)
		em = sig * conj(ep)      # ep = ± exp(i * (-m) * φ)
		dep_dφ = im *   m  * ep
		dem_dφ = im * (-m) * em
		for l in m:L
			p_div_sinθ = P[index_p(l,m)]
			@inbounds Y[index_y(l, -m)] = em * p_div_sinθ * S.sinθ
			@inbounds Y[index_y(l,  m)] = ep * p_div_sinθ * S.sinθ

			dp_dθ = dP[index_p(l,m)]
			@inbounds dY[index_y(l, -m)] = dspher_to_dcart(S, dem_dφ * p_div_sinθ, em * dp_dθ)
			@inbounds dY[index_y(l,  m)] = dspher_to_dcart(S, dep_dφ * p_div_sinθ, ep * dp_dθ)
		end
	end

	return Y, dY
end


# ---------------------- batched evaluation code 



"""
evaluate complex spherical harmonics
"""
function cYlm!(Y, L, S::AbstractVector{<: SphericalCoords}, P::AbstractMatrix, basis)
	nS = length(S) 
	@assert length(P) >= sizeP(L)
	@assert size(Y, 2) >= sizeY(L)
	@assert size(Y, 1) >= nS 
	t = acquire!(basis.tmp_t, nS)
	co = acquire!(basis.tmp_cos, nS)
	si = acquire!(basis.tmp_sin, nS)

	@inbounds begin 
		for i = 1:nS
			t[i] = 1 / sqrt(2) + im * 0
			co[i] = S[i].cosφ
			si[i] = S[i].sinφ
		end
	
		for l = 0:L 
			i_yl0 = index_y(l, 0)
			i_pl0 = index_p(l, 0)
			for i = 1:nS
				Y[i, i_yl0] = P[i, i_pl0] * t[i] 
			end
		end

		sig = 1
		for m in 1:L
			sig *= -1
			for i = 1:nS
				# t[i] =   exp(i *   m  * φ[i])   ... previously called ep 
				# and the previous em = ± exp(i * (-m) * φ) becomes sig * conj(t[i])
				t[i] *= co[i] + im * si[i]
			end

			for l in m:L
				i_plm = index_p(l,m)
				i_ylm⁺ = index_y(l,  m)
				i_ylm⁻ = index_y(l, -m)
				for i = 1:nS
					p = P[i, i_plm]
					Y[i, i_ylm⁻] = (sig * p) * conj(t[i])
					Y[i, i_ylm⁺] = t[i] * p  
				end
			end
		end
	end 

	return Y
end



"""
evaluate gradients of complex spherical harmonics
"""
function cYlm_ed!(Y, dY, L, S::AbstractVector{<: SphericalCoords}, 
					      P::AbstractMatrix, dP::AbstractMatrix, 
							basis::CYlmBasis)
   nS = length(S)
	@assert size(P, 2) >= sizeP(L)
	@assert size(P, 1) >= nS
	@assert size(dP, 2) >= sizeP(L)
	@assert size(dP, 1) >= nS
	@assert size(Y, 2) >= sizeY(L)
	@assert size(Y, 1) >= nS
	@assert size(dY, 2) >= sizeY(L)
	@assert size(dY, 1) >= nS
	ep = acquire!(basis.tmp_t, nS)
	co = acquire!(basis.tmp_cos, nS)
	si = acquire!(basis.tmp_sin, nS)

	# m = 0 case
	# ep = 1 / sqrt(2)
	fill!(ep, 1 / sqrt(2))

	@inbounds begin 

		for l = 0:L
			i_yl0 = index_y(l, 0)
			i_pl0 = index_p(l, 0)
			for i = 1:nS 
				Y[i, i_yl0] = P[i, i_pl0] * ep[i]
				dY[i, i_yl0] = dspher_to_dcart(S[i], 0.0, dP[i, i_pl0] * ep[i])
			end
		end

		sig = 1
		# ep_fact = S.cosφ + im * S.sinφ

		for m in 1:L
			sig *= -1
			# ep *= ep_fact            # ep =   exp(i *   m  * φ)
			for i = 1:nS
				ep[i] *= S[i].cosφ + im * S[i].sinφ
			end
			# em = sig * conj(ep)      # em = ± exp(i * (-m) * φ)
			# dep_dφ = im *   m  * ep
			# dem_dφ = im * (-m) * em
			for l in m:L
				i_plm = index_p(l,m)
				i_ylm⁺ = index_y(l,  m)
				i_ylm⁻ = index_y(l, -m)
				for i = 1:nS 
					em = sig * conj(ep[i])
					dep_dφ = im *   m  * ep[i]
					dem_dφ = im * (-m) * em

					p_div_sinθ = P[i, i_plm]
					Y[i, i_ylm⁻] = em * p_div_sinθ * S[i].sinθ
					Y[i, i_ylm⁺] = ep[i] * p_div_sinθ * S[i].sinθ

					dp_dθ = dP[i, i_plm]
					dY[i, i_ylm⁻] = dspher_to_dcart(S[i], dem_dφ * p_div_sinθ, em * dp_dθ)
					dY[i, i_ylm⁺] = dspher_to_dcart(S[i], dep_dφ * p_div_sinθ, ep[i] * dp_dθ)
				end
			end
		end

	end 

	return Y, dY
end
