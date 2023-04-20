export RYlmBasis 

struct RYlmBasis{T}
	alp::ALPolynomials{T}
   # ----------------------------
	pool::ArrayCache{T, 1}
   bpool::ArrayCache{T, 2}
	pool_d::ArrayCache{SVector{3, T}, 1}
   bpool_d::ArrayCache{SVector{3, T}, 2}
	tmp_s::TempArray{SphericalCoords{T}, 1}
	tmp_sin::TempArray{T, 1}
	tmp_cos::TempArray{T, 1}
	tmp_sinm::TempArray{T, 1}
	tmp_cosm::TempArray{T, 1}
end

RYlmBasis(maxL::Integer, T::Type=Float64) = 
      RYlmBasis(ALPolynomials(maxL, T))

RYlmBasis(alp::ALPolynomials{T}) where {T} = 
      RYlmBasis(alp, 
		          ArrayCache{T, 1}(), 
                ArrayCache{T, 2}(), 
                ArrayCache{SVector{3, T}, 1}(), 
                ArrayCache{SVector{3, T}, 2}(), 
					 TempArray{SphericalCoords{T}, 1}(),
					 TempArray{T, 1}(), 
					 TempArray{T, 1}(), 
					 TempArray{T, 1}(), 
					 TempArray{T, 1}() )


maxL(sh::RYlmBasis) = sh.alp.L

_valtype(sh::RYlmBasis{T}, x::AbstractVector{S}) where {T <: Real, S <: Real} = 
         promote_type(T, S)

Base.length(basis::RYlmBasis) = sizeY(maxL(basis))

Base.show(io::IO, basis::RYlmBasis) = 
      print(io, "RYlmBasis(L=$(maxL(basis)))")


# ---------------------- Interfaces

function evaluate(basis::RYlmBasis, x::AbstractVector{<: Real})
	Y = acquire!(basis.pool, length(basis), _valtype(basis, x))
	evaluate!(parent(Y), basis, x)
	return Y 
end

function evaluate!(Y, basis::RYlmBasis, x::AbstractVector{<: Real})
	L = maxL(basis)
	S = cart2spher(x) 
	P = evaluate(basis.alp, S)
	rYlm!(Y, maxL(basis), S, P)
	release!(P)
	return nothing 
end

function evaluate(basis::RYlmBasis, X::AbstractVector{<: AbstractVector})
	Y = acquire!(basis.bpool, (length(X), length(basis)))
	evaluate!(parent(Y), basis, X)
	return Y 
end

function evaluate!(Y, basis::RYlmBasis, 
						 X::AbstractVector{<: AbstractVector{<: Real}})
	L = maxL(basis)
	S = acquire!(basis.tmp_s, length(X))
	map!(cart2spher, S, X)
	P = evaluate(basis.alp, S)
	rYlm!(parent(Y), maxL(basis), S, parent(P), basis)
	release!(P)
	return nothing 
end


function evaluate_ed(basis::RYlmBasis, x::AbstractVector{<: Real})
	Y = acquire!(basis.pool, length(basis))
	dY = acquire!(basis.pool_d, length(basis))
	evaluate_ed!(parent(Y), parent(dY), basis, x)
	return Y, dY 
end

function evaluate_ed!(Y, dY, basis::RYlmBasis, 
						     x::AbstractVector{<: Real})
	L = maxL(basis)
	s = cart2spher(x)
	P, dP = _evaluate_ed(basis.alp, s)
	rYlm_ed!(parent(Y), parent(dY), maxL(basis), s, parent(P), parent(dP))
	release!(P)
	release!(dP)
	return nothing 
end


function evaluate_ed(basis::RYlmBasis, X::AbstractVector{<: AbstractVector{<: Real}})
	Y = acquire!(basis.bpool, (length(X), length(basis)))
	dY = acquire!(basis.bpool_d, (length(X), length(basis)))
	evaluate_ed!(parent(Y), parent(dY), basis, X)
	return Y, dY 
end

function evaluate_ed!(Y, dY, basis::RYlmBasis, 
						     X::AbstractVector{<: AbstractVector{<: Real}})
	L = maxL(basis)
	S = acquire!(basis.tmp_s, length(X))
	map!(cart2spher, S, X)
	P, dP = _evaluate_ed(basis.alp, S)
	rYlm_ed!(parent(Y), parent(dY), maxL(basis), S, parent(P), parent(dP), basis)
	release!(P)
	release!(dP)
	return nothing 
end


# -------------------- actual kernels 

"""
evaluate real spherical harmonics
"""
function rYlm!(Y, L, S, P::AbstractVector)
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
function rYlm_ed!(Y, dY, L, S, P, dP)
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


function rYlm!(Y::Matrix, L, S::AbstractVector, P::Matrix, basis::RYlmBasis)
   nX = length(S) 
	@assert size(P, 1) >= nX
   @assert size(P, 2) >= sizeP(L)
   @assert size(Y, 1) >= nX
	@assert size(Y, 2) >= sizeY(L)

   sinφ = acquire!(basis.tmp_sin, nX)
   cosφ = acquire!(basis.tmp_cos, nX)
   sinmφ = acquire!(basis.tmp_sinm, nX)
   cosmφ = acquire!(basis.tmp_cosm, nX)

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




function rYlm_ed!(Y::AbstractMatrix, dY::AbstractMatrix, L, S::AbstractVector, P, dP, basis)
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

   # @inbounds 
	begin 
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
         for i = 1:nX
            Y[i, i_yl0] = P[i, i_pl0] * oort2
				dY[i, i_yl0] = dspher_to_dcart(S[i], 0.0, dP[i, i_pl0] * oort2)
         end
      end

      for m in 1:L
         for i = 1:nX
            cmi = cosmφ[i]
            smi = sinmφ[i]
            cosmφ[i] = cmi * cosφ[i] - smi * sinφ[i]
            sinmφ[i] = smi * cosφ[i] + cmi * sinφ[i]
         end

         for l in m:L
            i_plm = index_p(l, m)
            i_ylm⁺ = index_y(l, m)
            i_ylm⁻ = index_y(l, -m)
            for i = 1:nX
					s = S[i] 
               p_div_sinθ = P[i, i_plm]
					p = p_div_sinθ * s.sinθ
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
					dY[i, i_ylm⁻] = dspher_to_dcart(s, a, b)
					dY[i, i_ylm⁺] = dspher_to_dcart(s, c, d)
            end
         end
      end

   end # inbounds 

	return nothing 
end