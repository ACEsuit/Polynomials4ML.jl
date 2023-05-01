"""
real spherical harmonics: SH = false: 

Yₗ⁰ = P̄ₗ⁰/√2
Yₗᵐ =  Re(P̄ₗᵐ(cosθ)/√2 exp(imφ))
Yₗ⁻ᵐ = -Im(P̄ₗᵐ(cosθ)/√2 exp(imφ))

solid harmonics: SH = true: 

Sₗ⁰ = √(4π/2l+1) rˡP̄ₗ⁰/√2
Sₗᵐ = (-1)ᵐ√(8π/2l+1) rˡ Re(P̄ₗᵐ(cosθ)/√2 exp(imφ))
Sₗ⁻ᵐ = (-1)ᵐ√(8π/2l+1) rˡIm(P̄ₗᵐ(cosθ)/√2 exp(imφ))
"""


struct RRlmBasis{T} <: RlmBasis
    alp::ALPolynomials{T}
   # ----------------------------
	pool::ArrayCache{T, 1}
   ppool::ArrayCache{T, 2}
	pool_d::ArrayCache{SVector{3, T}, 1}
   ppool_d::ArrayCache{SVector{3, T}, 2}
	tmp_s::TempArray{SphericalCoords{T}, 1}
	tmp_sin::TempArray{T, 1}
	tmp_cos::TempArray{T, 1}
	tmp_sinθ::TempArray{T, 1}
	tmp_cosθ::TempArray{T, 1}
	tmp_sinm::TempArray{T, 1}
	tmp_cosm::TempArray{T, 1}
	tmp_r::TempArray{T, 1}
	tmp_rL::TempArray{T, 2}
	tmp_aL::TempArray{T, 1}
	SH::Bool
end

RRlmBasis(maxL::Integer, SH::Bool, T::Type=Float64) = 
      RRlmBasis(ALPolynomials(maxL, T), SH)

RRlmBasis(alp::ALPolynomials{T}, SH::Bool) where {T} = 
      RRlmBasis(alp, 
		          ArrayCache{T, 1}(), 
                ArrayCache{T, 2}(), 
                ArrayCache{SVector{3, T}, 1}(), 
                ArrayCache{SVector{3, T}, 2}(), 
					 TempArray{SphericalCoords{T}, 1}(),
					 TempArray{T, 1}(), 
					 TempArray{T, 1}(), 
					 TempArray{T, 1}(), 
					 TempArray{T, 1}(), 
					 TempArray{T, 1}(), 
					 TempArray{T, 1}(), 
					 TempArray{T, 1}(),
					 TempArray{T, 2}(),
					 TempArray{T, 1}(),
					 SH)

_valtype(sh::RRlmBasis{T}, x::AbstractVector{S}) where {T <: Real, S <: Real} = 
         promote_type(T, S)

_gradtype(sh::RRlmBasis{T}, x::AbstractVector{S})  where {T <: Real, S <: Real} = 
		 SVector{3, Real}	

Base.show(io::IO, basis::RRlmBasis) = 
      print(io, "RRlmBasis(L=$(maxL(basis)))")

# ---------------------- evaluation interface code 
function evaluate!(Y, basis::RRlmBasis, X::AbstractVector{<: Real})
	L = maxL(basis)
	S = cart2spher(X;SH = basis.SH) 
	P = evaluate(basis.alp, S)
	rRlm!(Y, maxL(basis), S, P, basis)
	release!(P)
	return nothing 
end

function evaluate!(Y, basis::RRlmBasis, 
						 X::AbstractVector{<: AbstractVector{<: Real}})
	S = acquire!(basis.tmp_s, length(X))
	map!(X->cart2spher(X, SH = basis.SH), S, X) 
	P = evaluate(basis.alp, S)
	rRlm!(parent(Y), maxL(basis), S, parent(P), basis)
	release!(P)
	return nothing 
end

function evaluate_ed(basis::RRlmBasis, X::AbstractVector{<: Real})
	Y = acquire!(basis.pool, length(basis))
	dY = acquire!(basis.pool_d, length(basis))
	evaluate_ed!(parent(Y), parent(dY), basis, X)
	return Y, dY 
end

function evaluate_ed(basis::RRlmBasis, X::AbstractVector{<: AbstractVector{<: Real}})
	Y = acquire!(basis.ppool, (length(X), length(basis)))
	dY = acquire!(basis.ppool_d, (length(X), length(basis)))
	evaluate_ed!(parent(Y), parent(dY), basis, X)
	return Y, dY 
end

function evaluate_ed!(Y, dY, basis::RRlmBasis, 
						     X::AbstractVector{<: Real})
	S = cart2spher(X;SH = basis.SH) 
	P, dP = _evaluate_ed(basis.alp, S)
	rRlm_ed!(parent(Y), parent(dY), maxL(basis), S, parent(P), parent(dP), basis)
	release!(P)
	release!(dP)
	return nothing 
end

function evaluate_ed!(Y, dY, basis::RRlmBasis, 
						     X::AbstractVector{<: AbstractVector{<: Real}})
	S = acquire!(basis.tmp_s, length(X))
	map!(X->cart2spher(X, SH = basis.SH), S, X) 
	P, dP = _evaluate_ed(basis.alp, S)
	rRlm_ed!(parent(Y), parent(dY), maxL(basis), S, parent(P), parent(dP), basis)
	release!(P)
	release!(dP)
	return nothing 
end

# -------------------- actual kernels 

"""
evaluate real solid harmonics
"""
function rRlm!(Y, L, S::SphericalCoords, P::AbstractVector, basis::RRlmBasis)
	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)
    @assert abs(S.cosθ) <= 1.0

    oort2 = 1 / sqrt(2)
	rL = acquire!(basis.tmp_rL, (1, L+2))
	aL = acquire!(basis.tmp_aL, L+1)

	rL[1] = 1

	for l = 0:L
		if basis.SH
			aL[l+1] = sqrt(4*pi/(2*l+1))
		else
			aL[l+1] = 1
		end
		Y[index_y(l, 0)] = P[index_p(l, 0)] * oort2 * rL[l+1] * aL[l+1]
		rL[l+2] = rL[l+1] * S.r
	end

	sig⁺ = 1
	sig⁻ = -1
	ec = 1.0 + 0 * im
    ec_fact = S.cosφ + im * S.sinφ
	for m in 1:L
		if basis.SH
			sig⁺ *= -1
			sig⁻ = sig⁺
		end
		ec *= ec_fact
		for l in m:L
			p = P[index_p(l,m)] * rL[l+1] * aL[l+1]
			@inbounds Y[index_y(l,  m)] =  p * real(ec) * sig⁺
			@inbounds Y[index_y(l, -m)] =  p * imag(ec)  * sig⁻
		end
	end

	return nothing 
end


"""
evaluate gradients of real spherical harmonics
"""
function rRlm_ed!(Y, dY, L, S, P, dP, basis::RRlmBasis)
	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)
    @assert abs(S.cosθ) <= 1.0

    oort2 = 1 / sqrt(2)
	rL = acquire!(basis.tmp_rL, (1, L+2))
	aL = acquire!(basis.tmp_aL, L+1)
	fill!(rL,1.0)

	for l = 0:L
		if basis.SH
			aL[l+1] = sqrt(4*pi/(2*l+1))
		else
			aL[l+1] = 1
		end
		Y[index_y(l, 0)] = P[index_p(l, 0)] * oort2 * rL[l+1] * aL[l+1]
		dY[index_y(l, 0)] = dspher_to_dcart(S, l * Y[index_y(l, 0)], 0.0, rL[l+1] * aL[l+1] * dP[index_p(l, 0)] * oort2, SH = basis.SH)
		rL[l+2] = rL[l+1] * S.r
	end

	ec = 1.0 + 0 * im
    ec_fact = S.cosφ + im * S.sinφ
	sig⁺ = 1
	sig⁻ = -1
	for m in 1:L
		if basis.SH
			sig⁺ *= -1
			sig⁻ = sig⁺
		end
		ec *= ec_fact          # ec = exp(i * m  * φ) / sqrt(2)
		dec_dφ = im * m * ec
		for l in m:L
			p_div_sinθ = P[index_p(l,m)] * rL[l+1] * aL[l+1]
			p = p_div_sinθ * S.sinθ 
			dp_dθ = dP[index_p(l,m)] * rL[l+1] * aL[l+1]
			Y[index_y(l,  m)] =  p * real(ec) * sig⁺
			Y[index_y(l, -m)] =  p * imag(ec) * sig⁻
			dY[index_y(l,  m)] = dspher_to_dcart(S, l * Y[index_y(l, m)],  real(dec_dφ) * p_div_sinθ * sig⁺,
															      real(ec) * dp_dθ * sig⁺, SH = basis.SH)
			dY[index_y(l, -m)] = dspher_to_dcart(S, l * Y[index_y(l, -m)], imag(dec_dφ) * p_div_sinθ * sig⁻,
															    imag(ec) * dp_dθ * sig⁻, SH = basis.SH)
		end
	end

	return nothing 
end



# ---------------------- Batched evaluation
function rRlm!(Y::Matrix, L, S::AbstractVector{<: SphericalCoords}, P::AbstractMatrix, basis::RRlmBasis)
    nX = length(S) 
    @assert size(P, 1) >= nX
    @assert size(P, 2) >= sizeP(L)
    @assert size(Y, 1) >= nX
    @assert size(Y, 2) >= sizeY(L)
 
    sinφ = acquire!(basis.tmp_sin, nX)
    cosφ = acquire!(basis.tmp_cos, nX)
    sinmφ = acquire!(basis.tmp_sinm, nX)
    cosmφ = acquire!(basis.tmp_cosm, nX)
	r = acquire!(basis.tmp_r, nX)
    rL = acquire!(basis.tmp_rL, (nX, L+2))
	aL = acquire!(basis.tmp_aL, L+1)
 
    @inbounds begin 
        for i = 1:nX 
            sinφ[i] = S[i].sinφ
            cosφ[i] = S[i].cosφ
            sinmφ[i] = 0.0
            cosmφ[i] = 1.0
			r[i] = S[i].r
			rL[i,1] = 1.0
        end
 
        oort2 = 1 / sqrt(2)
        for l = 0:L
            if basis.SH
				aL[l+1] = sqrt(4*pi/(2*l+1))
			else
				aL[l+1] = 1
			end
            i_yl0 = index_y(l, 0)
            i_pl0 = index_p(l, 0)
            for i = 1:nX
                Y[i, i_yl0] = P[i, i_pl0] * oort2 * rL[i, l+1] * aL[l+1]
                rL[i, l+2] = rL[i, l+1] * r[i]
            end
        end
 
        sig⁺ = 1
		sig⁻ = -1
        for m in 1:L
            if basis.SH
				sig⁺ *= -1
				sig⁻ = sig⁺
			end
            @avx for i = 1:nX
                cmi = cosmφ[i]
                smi = sinmφ[i]
                cosmφ[i] = cmi * cosφ[i] - smi * sinφ[i]
                sinmφ[i] = smi * cosφ[i] + cmi * sinφ[i]
            end
 
            for l in m:L
                i_plm = index_p(l, m)
                i_ylm⁺ = index_y(l, m)
                i_ylm⁻ = index_y(l, -m)

                @avx for i = 1:nX
                    p = P[i, i_plm] * rL[i,l+1] * aL[l+1]
					Y[i, i_ylm⁻] =  p * sinmφ[i] * sig⁻
                    Y[i, i_ylm⁺] =  p * cosmφ[i] * sig⁺
                end
            end
        end
 
    end # inbounds 
 
    return nothing 
end




function rRlm_ed!(Y::AbstractMatrix, dY::AbstractMatrix, L, S::AbstractVector, P, dP, basis::RRlmBasis)
    nX = length(S) 
    @assert size(P, 1) >= nX
    @assert size(P, 2) >= sizeP(L)
    @assert size(dP, 1) >= nX
    @assert size(dP, 2) >= sizeP(L)
    @assert size(Y, 1) >= nX
    @assert size(Y, 2) >= sizeY(L)
    @assert size(dY, 1) >= nX
    @assert size(dY, 2) >= sizeY(L)
 
    sinφ = acquire!(basis.tmp_sin, nX)
    cosφ = acquire!(basis.tmp_cos, nX)
    sinmφ = acquire!(basis.tmp_sinm, nX)
    cosmφ = acquire!(basis.tmp_cosm, nX)
	r = acquire!(basis.tmp_r, nX)
    rL = acquire!(basis.tmp_rL, (nX, L+2))
	aL = acquire!(basis.tmp_aL, L+1)

    @inbounds begin 
        @simd ivdep for i = 1:nX 
            sinφ[i] = S[i].sinφ
            cosφ[i] = S[i].cosφ
            sinmφ[i] = 0.0
            cosmφ[i] = 1.0
			r[i] = S[i].r
			rL[i,1] = 1.0
        end
 
        oort2 = 1 / sqrt(2)
        for l = 0:L
            if basis.SH
				aL[l+1] = sqrt(4*pi/(2*l+1))
			else
				aL[l+1] = 1
			end
            i_yl0 = index_y(l, 0)
            i_pl0 = index_p(l, 0)
            @simd ivdep  for i = 1:nX
                Y[i, i_yl0] = P[i, i_pl0] * oort2 * rL[i, l+1] * aL[l+1]
                dY[i, i_yl0] = dspher_to_dcart(S[i], l * Y[i, i_yl0], 0.0, rL[i, l+1] * aL[l+1] * dP[i, i_pl0] * oort2, SH = basis.SH)
                rL[i, l+2] = rL[i, l+1] * r[i]
            end
        end
 
        sig⁺ = 1
		sig⁻ = -1
        for m in 1:L
            if basis.SH
				sig⁺ *= -1
				sig⁻ = sig⁺
			end
            @simd ivdep  for i = 1:nX
                cmi = cosmφ[i]
                smi = sinmφ[i]
                cosmφ[i] = cmi * cosφ[i] - smi * sinφ[i]
                sinmφ[i] = smi * cosφ[i] + cmi * sinφ[i]
            end
 
            for l in m:L
                i_plm = index_p(l, m)
                i_ylm⁺ = index_y(l, m)
                i_ylm⁻ = index_y(l, -m)
 
                @simd ivdep for i = 1:nX
                    p_div_sinθ = P[i, i_plm] * rL[i, l+1] * aL[l+1]
                    p = p_div_sinθ * S[i].sinθ
                    Y[i, i_ylm⁺] =  p * cosmφ[i] * sig⁺
					Y[i, i_ylm⁻] =  p * sinmφ[i] * sig⁻
					dp_dθ = dP[i, i_plm] * rL[i, l+1] * aL[l+1] 
                    a = m * cosmφ[i] * p_div_sinθ * sig⁻
                    b = sinmφ[i] * dp_dθ * sig⁻
                    c = -sig⁺ * m * sinmφ[i] * p_div_sinθ 
                    d = sig⁺ * cosmφ[i] * dp_dθ
                    dY[i, i_ylm⁻] = dspher_to_dcart(S[i], l * Y[i, i_ylm⁻], a, b, SH = basis.SH)
                    dY[i, i_ylm⁺] = dspher_to_dcart(S[i], l * Y[i, i_ylm⁺], c, d, SH = basis.SH)
                end
            end
        end
 
    end # inbounds 
 
    return nothing 
end

##  ---------------- Laplacian Implementation 

function _lap(basis::RRlmBasis, Y::AbstractVector) 
	ΔY = acquire!(basis.pool, length(Y))
	_lap!(parent(ΔY), basis, Y)
	return ΔY
end

function _lap!(ΔY, basis::RRlmBasis, Y::AbstractVector)
	if !basis.SH
		for i = 1:length(Y)
			l = idx2l(i)
			ΔY[i] = - Y[i] * l * (l+1)
		end
	else
		ΔY .= 0
	end
	return nothing 
end 

function _lap(basis::RRlmBasis, Y::AbstractMatrix) 
	ΔY = acquire!(basis.ppool, size(Y))
	_lap!(parent(ΔY), basis, Y)
	return ΔY
end

function _lap!(ΔY, basis::RRlmBasis, Y::AbstractMatrix) 
	@assert size(ΔY, 1) >= size(Y, 1)
	@assert size(ΔY, 2) >= size(Y, 2)
	@assert size(Y, 2) >= length(basis)
	nX = size(Y, 1)
	if !basis.SH
		@inbounds for l = 0:maxL(basis)
			λ = - l * (l+1)
			for m = -l:l
				i = index_y(l, m)
				@simd ivdep for j = 1:nX 
					ΔY[j, i] = λ * Y[j, i]
				end
			end
		end
	else
		ΔY .= 0
	end
	return nothing 
end 

function laplacian(basis::RRlmBasis, X)
	Y = evaluate(basis, X)
	ΔY = _lap(basis, Y)
	release!(Y)
	return ΔY
end

function laplacian!(ΔY, basis::RRlmBasis, X)
	Y = evaluate(basis, X)
	_lap!(ΔY, basis, Y)
	release!(Y)
	return ΔY
end


function eval_grad_laplace(basis::RRlmBasis, X)
	Y, dY = evaluate_ed(basis, X)
	ΔY = _lap(basis, Y)
	return Y, dY, ΔY
end
