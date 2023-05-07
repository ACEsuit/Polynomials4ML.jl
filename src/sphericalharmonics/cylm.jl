using ObjectPools: ArrayPool, FlexTempArray

"""
`CYlmBasis(maxL, T=Float64): `

Complex spherical harmonics; see tests to see how they are normalized, and  `idx2lm` on how they are ordered. The ordering is not guarenteed to be semver-stable.

The input variable is normally an `rr::SVector{3, T}`. This `rr` need not be normalized (i.e. on the unit sphere). The derivatives account for this, i.e. they are valid even when `norm(rr) != 1`.

* `maxL` : maximum degree of the spherical harmonics
* `T` : type used to store the coefficients for the associated legendre functions
"""
struct CYlmBasis{T} <: AbstractPoly4MLBasis
	alp::ALPolynomials{T}
   # ----------------------------
	tmp::ArrayPool{FlexTempArray}
end

CYlmBasis(maxL::Integer, T::Type=Float64) = 
      CYlmBasis(ALPolynomials(maxL, T))

CYlmBasis(alp::ALPolynomials{T}) where {T} = 
      CYlmBasis(alp, ArrayPool(FlexTempArray) )

Base.show(io::IO, basis::CYlmBasis) = 
      print(io, "CYlmBasis(L=$(maxL(basis)))")

_valtype(sh::CYlmBasis{T}, ::Type{<: StaticVector{3, S}}) where {T <: Real, S <: Real} = 
			Complex{promote_type(T, S)}


# ---------------------- FIO


# write_dict(SH::CYlmBasis{T}) where {T} =
# 		Dict("__id__" => "ACE_CYlmBasis",
# 			  "T" => write_dict(T),
# 			  "maxL" => maxL(SH))

# read_dict(::Val{:ACE_CYlmBasis}, D::Dict) =
# 		CYlmBasis(D["maxL"], read_dict(D["T"]))


		
# ---------------------- evaluation interface code 

_acqu_alp!(sym::Symbol, basis, S::SphericalCoords) = 
		acquire!(basis.tmp, sym, (length(basis.alp),), _valtype(basis.alp, S))

_acqu_alp!(sym::Symbol, basis, S::AbstractVector{<: SphericalCoords}) = 
		acquire!(basis.tmp, sym, (length(basis.alp),), _valtype(basis.alp, eltype(S)))

_acqu_P!(  basis, S) = _acqu_alp!(:alpP,   basis, S)
_acqu_dP!( basis, S) = _acqu_alp!(:alpdP,  basis, S)
_acqu_ddP!(basis, S) = _acqu_alp!(:alpddP, basis, S)

cart2spher(basis::CYlmBasis, x::AbstractVector{<: Real}) = cart2spher(x) 

function evaluate!(Y, basis::CYlmBasis, X)
	L = maxL(basis)
   S = cart2spher(basis, X)
	_P = _acqu_P!(basis, S)
	P = evaluate!(_P, basis.alp, S)
	cYlm!(Y, maxL(basis), S, P, basis)
	return Y
end


function evaluate_ed!(Y, dY, basis::CYlmBasis, X)
	L = maxL(basis)
	S = cart2spher(basis, X)
	_P, _dP = _acqu_P!(basis, S), _acqu_dP!(basis, S)
	P, dP = evaluate_ed!(_P, _dP, basis.alp, S)
	cYlm_ed!(Y, dY, maxL(basis), S, P, dP, basis)
	return Y, dY
end


# ---------------------- serial evaluation code 


"""
evaluate complex spherical harmonics
"""
function cYlm!(Y, L, S::SphericalCoords, P::AbstractVector, basis::CYlmBasis)
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
function cYlm_ed!(Y, dY, L, S::SphericalCoords, P, dP, basis::CYlmBasis)
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
function cYlm!(Y, L, S::AbstractVector{SphericalCoords{T}}, P::AbstractMatrix, basis) where {T} 
	nS = length(S) 
	@assert size(P, 1) >= nS 
	@assert size(P, 2) >= sizeP(L)
	@assert size(Y, 2) >= sizeY(L)
	@assert size(Y, 1) >= nS 

	t = acquire!(basis.tmp, :T, (nS,), Complex{T})
	co = acquire!(basis.tmp, :cos, (nS,), T)
	si = acquire!(basis.tmp, :sin, (nS,), T)

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
function cYlm_ed!(Y, dY, L, S::AbstractVector{SphericalCoords{T}}, 
					      P::AbstractMatrix, dP::AbstractMatrix, 
							basis::CYlmBasis) where {T} 
   nS = length(S)
	@assert size(P, 2) >= sizeP(L)
	@assert size(P, 1) >= nS
	@assert size(dP, 2) >= sizeP(L)
	@assert size(dP, 1) >= nS
	@assert size(Y, 2) >= sizeY(L)
	@assert size(Y, 1) >= nS
	@assert size(dY, 2) >= sizeY(L)
	@assert size(dY, 1) >= nS

	ep = acquire!(basis.tmp, :T, (nS,), Complex{T})
	co = acquire!(basis.tmp, :cos, (nS,), T)
	si = acquire!(basis.tmp, :sin, (nS,), T)

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
