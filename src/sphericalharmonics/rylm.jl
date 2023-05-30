using ChainRulesCore
using HyperDualNumbers: Hyper

export RYlmBasis 

"""
`RYlmBasis(maxL, T=Float64): `

Real spherical harmonics; see tests to see how they are normalized, and  `idx2lm` on how they are ordered. The ordering is not guarenteed to be semver-stable.

The input variable is normally an `rr::SVector{3, T}`. This `rr` need not be normalized (i.e. on the unit sphere). The derivatives account for this, i.e. they are valid even when `norm(rr) != 1`.

* `maxL` : maximum degree of the spherical harmonics
* `T` : type used to store the coefficients for the associated legendre functions
"""
struct RYlmBasis{T} <: AbstractPoly4MLBasis
	alp::ALPolynomials{T}
	@reqfields
end

RYlmBasis(maxL::Integer, T::Type=Float64) = 
      RYlmBasis(ALPolynomials(maxL, T))

RYlmBasis(alp::ALPolynomials{T}) where {T} = 
      RYlmBasis(alp, _make_reqfields()...)

_valtype(sh::RYlmBasis{T}, ::Type{<: StaticVector{3, S}}) where {T <: Real, S <: Real} = 
		promote_type(T, S)

_valtype(sh::RYlmBasis{T}, ::Type{<: StaticVector{3, Hyper{S}}}) where {T <: Real, S <: Real} = 
		promote_type(T, S)

Base.show(io::IO, basis::RYlmBasis) = 
      print(io, "RYlmBasis(L=$(maxL(basis)))")		

# ---------------------- Interfaces

function evaluate!(Y::AbstractArray, basis::RYlmBasis, X)
	L = maxL(basis)
    S = cart2spher(basis, X)
	_P = _acqu_P!(basis, S)
	P = evaluate!(_P, basis.alp, S)
	rYlm!(Y, maxL(basis), S, parent(P), basis)
	return Y
end


function evaluate_ed!(Y::AbstractArray, dY::AbstractArray, basis::RYlmBasis, X)
	L = maxL(basis)
	S = cart2spher(basis, X)
	_P, _dP = _acqu_P!(basis, S), _acqu_dP!(basis, S)
	P, dP = evaluate_ed!(_P, _dP, basis.alp, S)
	rYlm_ed!(Y, dY, maxL(basis), S, parent(P), parent(dP), basis)
	return Y, dY
end


# -------------------- actual kernels 

"""
evaluate real spherical harmonics
"""
function rYlm!(Y, L, S, P::AbstractVector, basis::RYlmBasis)
	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)
   @assert abs(S.cosθ) <= 1.0

   oort2 = 1 / sqrt(2)
	for l = 0:L
		Y[index_y(l, 0)] = P[index_p(l, 0)] * oort2
	end

   sig = 1
	ec = 1.0 + 0 * im
   ec_fact = S.cosφ + im * S.sinφ
	for m in 1:L
		sig *= -1                # sig = (-1)^m
		ec *= ec_fact            # ec = exp(i * m  * φ) / sqrt(2)
		# cYlm = p * ec,    (also cYl{-m} = sig * p * conj(ec)), but not needed)
		# rYlm    =  Re(cYlm)
		# rYl{-m} = -Im(cYlm)
		for l in m:L
			p = P[index_p(l,m)]
			@inbounds Y[index_y(l, -m)] = -p * imag(ec)
			@inbounds Y[index_y(l,  m)] =  p * real(ec)
		end
	end

	return nothing 
end


"""
evaluate gradients of real spherical harmonics
"""
function rYlm_ed!(Y, dY, L, S::SphericalCoords, P, dP, basis::RYlmBasis)
	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)
   @assert abs(S.cosθ) <= 1.0

   oort2 = 1 / sqrt(2)
	for l = 0:L
		Y[index_y(l, 0)] = P[index_p(l, 0)] * oort2
		dY[index_y(l, 0)] = dspher_to_dcart(S, 0.0, dP[index_p(l, 0)] * oort2)
	end

   sig = 1
	ec = 1.0 + 0 * im
   ec_fact = S.cosφ + im * S.sinφ
	drec_dφ = 0.0
	for m in 1:L
		sig *= -1                # sig = (-1)^m
		ec *= ec_fact            # ec = exp(i * m  * φ) / sqrt(2)
		dec_dφ = im * m * ec

		# cYlm = p * ec,    (also cYl{-m} = sig * p * conj(ec)), but not needed)
		# rYlm    =  Re(cYlm)
		# rYl{-m} = -Im(cYlm)
		for l in m:L
			p_div_sinθ = P[index_p(l,m)]
			p = p_div_sinθ * S.sinθ
			Y[index_y(l, -m)] = -p * imag(ec)
			Y[index_y(l,  m)] =  p * real(ec)

			dp_dθ = dP[index_p(l,m)]
			dY[index_y(l, -m)] = dspher_to_dcart(S, - imag(dec_dφ) * p_div_sinθ,
															    - imag(ec) * dp_dθ)
			dY[index_y(l,  m)] = dspher_to_dcart(S,   real(dec_dφ) * p_div_sinθ,
															      real(ec) * dp_dθ)
		end
	end

	return nothing 
end



# ---------------------- Batched evaluation


function rYlm!(Y::AbstractMatrix, L, S::AbstractVector{SphericalCoords{T}}, 
				   P::AbstractMatrix, basis::RYlmBasis) where {T} 
   nX = length(S) 
	@assert size(P, 1) >= nX
   @assert size(P, 2) >= sizeP(L)
   @assert size(Y, 1) >= nX
	@assert size(Y, 2) >= sizeY(L)

   sinφ = acquire!(basis.tmp, :sin, (nX,), T)
   cosφ = acquire!(basis.tmp, :cos, (nX,), T)
   sinmφ = acquire!(basis.tmp, :sinm, (nX,), T)
   cosmφ = acquire!(basis.tmp, :cosm, (nX,), T)

   @inbounds begin 
      for i = 1:nX 
         sinφ[i] = S[i].sinφ
         cosφ[i] = S[i].cosφ
         sinmφ[i] = 0.0
         cosmφ[i] = 1.0
      end

      oort2 = 1 / sqrt(2)
      for l = 0:L
         i_yl0 = index_y(l, 0)
         i_pl0 = index_p(l, 0)
         @avx for i = 1:nX
            Y[i, i_yl0] = P[i, i_pl0] * oort2
         end
      end

      for m in 1:L
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
               p = P[i, i_plm]
               Y[i, i_ylm⁺] =  p * cosmφ[i]
               Y[i, i_ylm⁻] = -p * sinmφ[i]
            end
         end
      end

   end # inbounds 

	return nothing 
end




function rYlm_ed!(Y::AbstractMatrix, dY::AbstractMatrix, L, 
					  S::AbstractVector{SphericalCoords{T}}, P, dP, basis) where {T} 
   nX = length(S) 
	@assert size(P, 1) >= nX
   @assert size(P, 2) >= sizeP(L)
	@assert size(dP, 1) >= nX
   @assert size(dP, 2) >= sizeP(L)
   @assert size(Y, 1) >= nX
	@assert size(Y, 2) >= sizeY(L)
   @assert size(dY, 1) >= nX
	@assert size(dY, 2) >= sizeY(L)

   sinφ = acquire!(basis.tmp, :sin, (nX,), T)
   cosφ = acquire!(basis.tmp, :cos, (nX,), T)
   sinmφ = acquire!(basis.tmp, :sinm, (nX,), T)
   cosmφ = acquire!(basis.tmp, :cosm, (nX,), T)
   sinθ = acquire!(basis.tmp, :sinθ, (nX,), T)
   cosθ = acquire!(basis.tmp, :cosθ, (nX,), T)

   @inbounds begin 
      @simd ivdep for i = 1:nX 
         sinφ[i] = S[i].sinφ
         cosφ[i] = S[i].cosφ
			sinθ[i] = S[i].sinθ
			cosθ[i] = S[i].cosθ
         sinmφ[i] = 0.0
         cosmφ[i] = 1.0
      end

      oort2 = 1 / sqrt(2)
      for l = 0:L
         i_yl0 = index_y(l, 0)
         i_pl0 = index_p(l, 0)
         @simd ivdep  for i = 1:nX
            Y[i, i_yl0] = P[i, i_pl0] * oort2
				dY[i, i_yl0] = dspher_to_dcart(S[i], 0.0, dP[i, i_pl0] * oort2)
         end
      end

      for m in 1:L
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
               p_div_sinθ = P[i, i_plm]
					p = p_div_sinθ * sinθ[i]
               Y[i, i_ylm⁺] =  p * cosmφ[i]
               Y[i, i_ylm⁻] = -p * sinmφ[i]
					#
					# ec = exp(i * m  * φ) / sqrt(2)
					# 	dec_dφ = im * m * ec
					#  imag(dec_dφ) = m * real(ec) = m * cos(m * φ) / sqrt(2)
					#  imag(ex) = sin(m * φ) / sqrt(2)
					#  real(dec_dφ) = -m * imag(ec) = -m * sin(m * φ) / sqrt(2)
					#  real(ec) = cos(m * φ) / sqrt(2)
					#
					dp_dθ = dP[i, i_plm]
					a = - m * cosmφ[i] * p_div_sinθ
					b = - sinmφ[i] * dp_dθ
					c = - m * sinmφ[i] * p_div_sinθ
					d = cosmφ[i] * dp_dθ
					# dY[i, i_ylm⁻] = dspher_to_dcart(1.0, sinφ[i], cosφ[i], sinθ[i], cosθ[i], a, b)
					# dY[i, i_ylm⁺] = dspher_to_dcart(1.0, sinφ[i], cosφ[i], sinθ[i], cosθ[i], c, d)
					dY[i, i_ylm⁻] = dspher_to_dcart(S[i], a, b)
					dY[i, i_ylm⁺] = dspher_to_dcart(S[i], c, d)
            end
         end
      end

   end # inbounds 

	return nothing 
end


##  ---------------- Laplacian Implementation -- prototype 

function _lap(basis::RYlmBasis, Y::AbstractVector) 
	ΔY = Vector{eltype(Y)}(undef, length(Y))
	_lap!(ΔY, basis, Y)
	return ΔY
end

function _lap!(ΔY, basis::RYlmBasis, Y::AbstractVector) 
	for i = 1:length(Y)
		l = idx2l(i)
		ΔY[i] = - Y[i] * l * (l+1)
	end
	return nothing 
end 

function _lap(basis::RYlmBasis, Y::AbstractMatrix) 
	ΔY = acquire!(basis.pool, :ΔY, size(Y), eltype(Y))
	_lap!(parent(ΔY), basis, Y)
	return ΔY
end

function _lap!(ΔY, basis::RYlmBasis, Y::AbstractMatrix) 
	@assert size(ΔY, 1) >= size(Y, 1)
	@assert size(ΔY, 2) >= size(Y, 2)
	@assert size(Y, 2) >= length(basis)
	nX = size(Y, 1)
	@inbounds for l = 0:maxL(basis)
		λ = - l * (l+1)
		for m = -l:l
			i = index_y(l, m)
			@simd ivdep for j = 1:nX 
				ΔY[j, i] = λ * Y[j, i]
			end
		end
	end
	return nothing 
end 

function laplacian(basis::RYlmBasis, X)
	Y = evaluate(basis, X)
	ΔY = _lap(basis, Y)
	release!(Y)
	return ΔY
end

function laplacian!(ΔY, basis::RYlmBasis, X)
	Y = evaluate(basis, X)
	_lap!(ΔY, basis, Y)
	release!(Y)
	return ΔY
end


function eval_grad_laplace(basis::RYlmBasis, X)
	Y, dY = evaluate_ed(basis, X)
	ΔY = _lap(basis, Y)
	return Y, dY, ΔY
end

# Placeholder for now
function ChainRulesCore.rrule(::typeof(evaluate), basis::RYlmBasis, X)
	A  = evaluate(basis, X)
	∂X = similar(X)
   	dX = evaluate_ed(basis, X)[2]
	function pb(∂A)
		@assert size(∂A) == (length(X), length(basis))
		for i = 1:length(X)
            ∂X[i] = sum([∂A[i,j] * dX[i,j] for j = 1:length(dX[i,:])])
        end
		return NoTangent(), NoTangent(), ∂X
	end
	return A, pb
end