

using StaticArrays, LinearAlgebra, LoopVectorization

export CYlmBasis, RYlmBasis, RRlmBasis, CRlmBasis, RRlmBasis 


# --------------------------------------------------------
#     Coordinates
# --------------------------------------------------------

"""
`struct SphericalCoords` : a simple datatype storing spherical coordinates
of a point (x,y,z) in the format `(r, cosφ, sinφ, cosθ, sinθ)`.
Use `spher2cart` and `cart2spher` to convert between cartesian and spherical
coordinates.
"""
struct SphericalCoords{T}
	r::T
	cosφ::T
	sinφ::T
	cosθ::T
	sinθ::T
end

spher2cart(S::SphericalCoords) = S.r * SVector(S.cosφ*S.sinθ, S.sinφ*S.sinθ, S.cosθ)

function cart2spher(R::AbstractVector) # ; SH = true)
	@assert length(R) == 3
	r = norm(R)
	φ = atan(R[2], R[1])
	sinφ, cosφ = sincos(φ)
	cosθ = R[3] / r
	sinθ = sqrt(R[1]^2+R[2]^2) / r
	# if SH
	# 	return SphericalCoords(r, cosφ, sinφ, cosθ, sinθ)
	# else
	# return SphericalCoords(1.0, cosφ, sinφ, cosθ, sinθ)
	# end
	return SphericalCoords(r, cosφ, sinφ, cosθ, sinθ)
end

SphericalCoords(φ, θ) = SphericalCoords(1.0, cos(φ), sin(φ), cos(θ), sin(θ))
SphericalCoords(r, φ, θ) = SphericalCoords(r, cos(φ), sin(φ), cos(θ), sin(θ))

"""
convert a gradient with respect to spherical coordinates to a gradient
with respect to cartesian coordinates
"""
function dspher_to_dcart(S, f_φ_div_sinθ, f_θ)
	r = S.r
   return SVector( - (S.sinφ * f_φ_div_sinθ) + (S.cosφ * S.cosθ * f_θ),
			            (S.cosφ * f_φ_div_sinθ) + (S.sinφ * S.cosθ * f_θ),
			 			                                 - (   S.sinθ * f_θ) ) / (r+eps(r))
end

dspher_to_dcart(r, sinφ, cosφ, sinθ, cosθ, f_φ_div_sinθ, f_θ) = 
   	SVector( - (sinφ * f_φ_div_sinθ) + (cosφ * cosθ * f_θ),
			            (cosφ * f_φ_div_sinθ) + (sinφ * cosθ * f_θ),
			 			                                 - (   sinθ * f_θ) ) / (r+eps(r))

function dspher_to_dcart(S, f_r_times_r, f_φ_div_sinθ, f_θ) # ; SH = true)
	r = S.r
	# if SH
	return SVector((S.sinθ * S.cosφ * f_r_times_r) - (S.sinφ * f_φ_div_sinθ) + (S.cosφ * S.cosθ * f_θ),
					(S.sinθ * S.sinφ * f_r_times_r) + (S.cosφ * f_φ_div_sinθ) + (S.sinφ * S.cosθ * f_θ),
							(S.cosθ * f_r_times_r) - (S.sinθ * f_θ))/ (r+eps(r))
	# else
	# 	return SVector( - (S.sinφ * f_φ_div_sinθ) + (S.cosφ * S.cosθ * f_θ),
	# 		            (S.cosφ * f_φ_div_sinθ) + (S.sinφ * S.cosθ * f_θ),
	# 		 			                                 - (   S.sinθ * f_θ) ) / r
	# end
end

dspher_to_dcart(r, sinφ, cosφ, sinθ, cosθ, f_r_times_r, f_φ_div_sinθ, f_θ) = 
   	SVector( (sinθ * cosφ * f_r_times_r) - (sinφ * f_φ_div_sinθ) + (cosφ * cosθ * f_θ),
	   				(sinθ * sinφ * f_r_times_r) + (cosφ * f_φ_div_sinθ) + (sinφ * cosθ * f_θ),
					   (cosθ * f_r_times_r) - (sinθ * f_θ)) / (r+eps(r)) 

include("alp.jl")
include("cylm.jl")
include("rylm.jl")

include("crlm.jl")
include("rrlm.jl")

const XlmBasis = Union{RYlmBasis, CYlmBasis, CRlmBasis, RRlmBasis}

"""
max L degree for which the alp coefficients have been precomputed
"""
maxL(basis::XlmBasis) = basis.alp.L

Base.length(basis::XlmBasis) = sizeY(maxL(basis))

natural_indices(basis::XlmBasis) = 
		[ NamedTuple{(:l, :m)}(idx2lm(i)) for i = 1:length(basis) ]

degree(basis::XlmBasis, b::NamedTuple) = b.l 

# ---------------------------- Auxiliary functions 

function rand_sphere() 
	r = @SVector randn(3)
	return r / norm(r)
end

# ---------------------- evaluation interface code 
function evaluate(basis::XlmBasis, x::AbstractVector{<: Real})
	Y = acquire!(basis.pool, length(basis), _valtype(basis, x))
	evaluate!(parent(Y), basis, x)
	return Y 
end

function evaluate(basis::XlmBasis, X::AbstractVector{<: AbstractVector})
	Y = acquire!(basis.ppool, (length(X), length(basis)))
	evaluate!(parent(Y), basis, X)
	return Y 
end

