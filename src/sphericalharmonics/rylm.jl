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
	return Y
end

function evaluate(basis::RYlmBasis, X::AbstractVector{<: AbstractVector})
	Y = acquire!(basis.bpool, (length(X), length(basis)))
	evaluate!(parent(Y), basis, X)
	return Y 
end

function evaluate!(Y, basis::RYlmBasis, 
						 X::AbstractVector{<: AbstractVector})
	L = maxL(basis)
	S = acquire!(basis.tmp_s, length(X))
	map!(cart2spher, S, X)
	P = evaluate(basis.alp, S)
	rYlm!(parent(Y), maxL(basis), S, parent(P), basis)
	release!(P)
	return Y
end


"""
evaluate real spherical harmonics
"""
function rYlm!(Y, L, S, P::AbstractVector)
	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)
   @assert abs(S.cos??) <= 1.0

   oort2 = 1 / sqrt(2)
	for l = 0:L
		Y[index_y(l, 0)] = P[index_p(l, 0)] * oort2
	end

   sig = 1
	ec = 1.0 + 0 * im
   ec_fact = S.cos?? + im * S.sin??
	for m in 1:L
		sig *= -1                # sig = (-1)^m
		ec *= ec_fact            # ec = exp(i * m  * ??) / sqrt(2)
		# cYlm = p * ec,    (also cYl{-m} = sig * p * conj(ec)), but not needed)
		# rYlm    =  Re(cYlm)
		# rYl{-m} = -Im(cYlm)
		for l in m:L
			p = P[index_p(l,m)]
			@inbounds Y[index_y(l, -m)] = -p * imag(ec)
			@inbounds Y[index_y(l,  m)] =  p * real(ec)
		end
	end

	return Y
end


"""
evaluate gradients of real spherical harmonics
"""
function rYlm_d!(Y, dY, L, S, P, dP)
	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)
   @assert abs(S.cos??) <= 1.0

   oort2 = 1 / sqrt(2)
	for l = 0:L
		Y[index_y(l, 0)] = P[index_p(l, 0)] * oort2
		dY[index_y(l, 0)] = dspher_to_dcart(S, 0.0, dP[index_p(l, 0)] * oort2)
	end

   sig = 1
	ec = 1.0 + 0 * im
   ec_fact = S.cos?? + im * S.sin??
	drec_d?? = 0.0
	for m in 1:L
		sig *= -1                # sig = (-1)^m
		ec *= ec_fact            # ec = exp(i * m  * ??) / sqrt(2)
		dec_d?? = im * m * ec

		# cYlm = p * ec,    (also cYl{-m} = sig * p * conj(ec)), but not needed)
		# rYlm    =  Re(cYlm)
		# rYl{-m} = -Im(cYlm)
		for l in m:L
			p_div_sin?? = P[index_p(l,m)]
			p = p_div_sin?? * S.sin??
			Y[index_y(l, -m)] = -p * imag(ec)
			Y[index_y(l,  m)] =  p * real(ec)

			dp_d?? = dP[index_p(l,m)]
			dY[index_y(l, -m)] = dspher_to_dcart(S, - imag(dec_d??) * p_div_sin??,
															    - imag(ec) * dp_d??)
			dY[index_y(l,  m)] = dspher_to_dcart(S,   real(dec_d??) * p_div_sin??,
															      real(ec) * dp_d??)
		end
	end

	return dY
end



# ---------------------- Batched evaluation

"""
evaluate real spherical harmonics
"""
function rYlm!(Y::Matrix, L, S::AbstractVector, P::Matrix, basis::RYlmBasis)
   nX = length(S) 
	@assert size(P, 1) >= nX
   @assert size(P, 2) >= sizeP(L)
   @assert size(Y, 1) >= nX
	@assert size(Y, 2) >= sizeY(L)

   sin?? = acquire!(basis.tmp_sin, nX)
   cos?? = acquire!(basis.tmp_cos, nX)
   sinm?? = acquire!(basis.tmp_sinm, nX)
   cosm?? = acquire!(basis.tmp_cosm, nX)

   @inbounds begin 
      for i = 1:nX 
         sin??[i] = S[i].sin??
         cos??[i] = S[i].cos??
         sinm??[i] = 0.0
         cosm??[i] = 1.0
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
            cmi = cosm??[i]
            smi = sinm??[i]
            cosm??[i] = cmi * cos??[i] - smi * sin??[i]
            sinm??[i] = smi * cos??[i] + cmi * sin??[i]
         end

         for l in m:L
            i_plm = index_p(l, m)
            i_ylm??? = index_y(l, m)
            i_ylm??? = index_y(l, -m)
            @avx for i = 1:nX
               p = P[i, i_plm]
               Y[i, i_ylm???] =  p * cosm??[i]
               Y[i, i_ylm???] = -p * sinm??[i]
            end
         end
      end

   end # inbounds 

	return Y
end


