"""
complex spherical harmonics: 

Yₗ⁰(θ, φ) = P̄ₗ⁰(cosθ)/√2
Yₗᵐ(θ, φ) = P̄ₗᵐ(cosθ)exp(imφ)/√2
Yₗ⁻ᵐ(θ, φ) = (-1)ᵐ P̄ₗᵐ(cosθ)/√2 exp(-imφ)

solid harmonics:

γₗᵐ(r, θ, φ) = rˡYₗᵐ(θ, φ)
"""
struct CRlmBasis{T} <: AbstractPoly4MLBasis
    alp::ALPolynomials{T}
	 @reqfields
end

CRlmBasis(maxL::Integer, T::Type=Float64) = 
      CRlmBasis(ALPolynomials(maxL, T))

CRlmBasis(alp::ALPolynomials{T}) where {T} = 
        CRlmBasis(alp, _make_reqfields()...) 

Base.show(io::IO, basis::CRlmBasis) = 
      print(io, "CRlmBasis(L=$(maxL(basis)))")

_valtype(sh::CRlmBasis{T}, ::Type{<: StaticVector{3, S}}) where {T <: Real, S <: Real} = 
		Complex{promote_type(T, S)}


# ---------------------- evaluation interface code 

function evaluate!(Y::AbstractArray, basis::CRlmBasis, X)
	L = maxL(basis)
   S = cart2spher(basis, X)
	_P = _acqu_P!(basis, S)
	P = evaluate!(_P, basis.alp, S)
	cRlm!(Y, maxL(basis), S, P, basis)
	return Y
end


function evaluate_ed!(Y::AbstractArray, dY::AbstractArray, basis::CRlmBasis, X)
	L = maxL(basis)
	S = cart2spher(basis, X)
	_P, _dP = _acqu_P!(basis, S), _acqu_dP!(basis, S)
	P, dP = evaluate_ed!(_P, _dP, basis.alp, S)
	cRlm_ed!(Y, dY, maxL(basis), S, P, dP, basis)
	return Y, dY
end

# -------------------- actual kernels 

"""
evaluate complex spherical harmonics
"""
function cRlm!(Y, L, S::SphericalCoords{T}, P::AbstractVector, basis::CRlmBasis) where {T}
	@assert length(P) >= sizeP(L) # evaluate(basis.alp, S)
	@assert length(Y) >= sizeY(L)
    @assert abs(S.cosθ) <= 1.0
	ep = 1 / sqrt(2) + im * 0 
    rL = acquire!(basis.tmp, :rL, (L+2,), T)
	rL[1] = 1.0

	for l = 0:L
		Y[index_y(l, 0)] =  P[index_p(l, 0)] * ep * rL[l+1]
		rL[l+2] = rL[l+1] * S.r
	end

    sig = 1
    ep_fact = S.cosφ + im * S.sinφ
	for m in 1:L
		sig *= -1
		ep *= ep_fact            # ep =   exp(i *   m  * φ)
		em = sig * conj(ep)      # em = ± exp(i * (-m) * φ)
		for l in m:L
			p = P[index_p(l,m)] * rL[l+1]
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
function cRlm_ed!(Y, dY, L, S::SphericalCoords{T}, P, dP, basis::CRlmBasis) where {T} 
	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)
	@assert length(dY) >= sizeY(L)

	# m = 0 case
	ep = 1 / sqrt(2)
	rL = acquire!(basis.tmp, :rL, (L+2,), T)
	rL[1] = 1.0

	for l = 0:L
		Y[index_y(l, 0)] =  P[index_p(l, 0)] * ep * rL[l+1]
		# f_r_times_r
		dY[index_y(l, 0)] = dspher_to_dcart(S, l * Y[index_y(l, 0)], 0.0, rL[l+1] * dP[index_p(l, 0)] * ep)
		rL[l+2] = rL[l+1] * S.r
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
			p_div_sinθ = P[index_p(l,m)] * rL[l+1]
			@inbounds Y[index_y(l, -m)] = em * p_div_sinθ * S.sinθ  
			@inbounds Y[index_y(l,  m)] = ep * p_div_sinθ * S.sinθ 
			dp_dθ = dP[index_p(l,m)] * rL[l+1]
			# f_r_times_r
			@inbounds dY[index_y(l, -m)] = dspher_to_dcart(S, l * Y[index_y(l, -m)], dem_dφ * p_div_sinθ , em * dp_dθ)
			@inbounds dY[index_y(l,  m)] = dspher_to_dcart(S, l * Y[index_y(l, m)], dep_dφ * p_div_sinθ, ep * dp_dθ)
		end
	end

	return Y, dY
end
# ---------------------- batched evaluation code 



"""
evaluate complex spherical harmonics
"""
function cRlm!(Y, L, S::AbstractVector{SphericalCoords{T}}, P::AbstractMatrix, basis::CRlmBasis) where {T} 
	nS = length(S) 
	@assert length(P) >= sizeP(L)
	@assert size(Y, 2) >= sizeY(L)
	@assert size(Y, 1) >= nS 
	t = acquire!(basis.tmp, :t, (nS,), Complex{T})
	co = acquire!(basis.tmp, :cos, (nS,), T)
	si = acquire!(basis.tmp, :sin, (nS,), T)
	r = acquire!(basis.tmp, :r, (nS,), T)
   rL = acquire!(basis.tmp, :rL, (nS, L+2), T)
	@inbounds begin 
		for i = 1:nS
			t[i] = 1 / sqrt(2) + im * 0
			co[i] = S[i].cosφ
			si[i] = S[i].sinφ
			r[i] = S[i].r
			rL[i,1] = 1
		end
	
		for l = 0:L 
			i_yl0 = index_y(l, 0)
			i_pl0 = index_p(l, 0)
			for i = 1:nS
				Y[i, i_yl0] = P[i, i_pl0] * t[i] * rL[i, l+1]
                rL[i, l+2] = rL[i, l+1] * r[i]
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
					p = P[i, i_plm] * rL[i, l+1]
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
function cRlm_ed!(Y, dY, L, S::AbstractVector{SphericalCoords{T}}, 
					      P::AbstractMatrix, dP::AbstractMatrix, 
							basis::CRlmBasis) where {T} 
    nS = length(S)
	@assert size(P, 2) >= sizeP(L)
	@assert size(P, 1) >= nS
	@assert size(dP, 2) >= sizeP(L)
	@assert size(dP, 1) >= nS
	@assert size(Y, 2) >= sizeY(L)
	@assert size(Y, 1) >= nS
	@assert size(dY, 2) >= sizeY(L)
	@assert size(dY, 1) >= nS
	
	ep = acquire!(basis.tmp, :t, (nS,), Complex{T})
	r = acquire!(basis.tmp, :r, (nS,), T)
	rL = acquire!(basis.tmp, :rL, (nS, L+2), T)

	@inbounds begin 
		for i = 1:nS
			ep[i] = 1 / sqrt(2)
			r[i] = S[i].r
			rL[i,1] = 1
		end
	

		for l = 0:L
			i_yl0 = index_y(l, 0)
			i_pl0 = index_p(l, 0)
			for i = 1:nS 
				Y[i, i_yl0] = P[i, i_pl0] * ep[i] * rL[i, l+1]
				dY[i, i_yl0] = dspher_to_dcart(S[i], l * Y[i, i_yl0], 0.0, rL[i, l+1] *  dP[i, i_pl0] * ep[i])
				rL[i, l+2] = rL[i, l+1] * r[i]
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

					p_div_sinθ = P[i, i_plm] * rL[i, l+1]
					Y[i, i_ylm⁻] = em * p_div_sinθ * S[i].sinθ
					Y[i, i_ylm⁺] = ep[i] * p_div_sinθ * S[i].sinθ

					dp_dθ = dP[i, i_plm] * rL[i, l+1]
					dY[i, i_ylm⁻] = dspher_to_dcart(S[i], l * Y[i, i_ylm⁻], dem_dφ * p_div_sinθ, em * dp_dθ)
					dY[i, i_ylm⁺] = dspher_to_dcart(S[i], l * Y[i, i_ylm⁺], dep_dφ * p_div_sinθ, ep[i] * dp_dθ)
				end
			end
		end

	end 

	return Y, dY
end