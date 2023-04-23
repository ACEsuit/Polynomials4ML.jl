

using StaticArrays, LinearAlgebra

export CYlmBasis, RYlmBasis, CRlmBasis, RRlmBasis


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

function cart2spher(R::AbstractVector)
	@assert length(R) == 3
	r = norm(R)
	φ = atan(R[2], R[1])
	sinφ, cosφ = sincos(φ)
	cosθ = R[3] / r
	sinθ = sqrt(R[1]^2+R[2]^2) / r
	return SphericalCoords(r, cosφ, sinφ, cosθ, sinθ)
end

SphericalCoords(φ, θ) = SphericalCoords(1.0, cos(φ), sin(φ), cos(θ), sin(θ))
SphericalCoords(r, φ, θ) = SphericalCoords(r, cos(φ), sin(φ), cos(θ), sin(θ))

"""
convert a gradient with respect to spherical coordinates to a gradient
with respect to cartesian coordinates

∂r = (sinθcosφ, sinθsinφ, cosθ)
∂φ = (-sinφ/rsinθ, cosφ/rsinθ, 0)
∂θ = (cosφcosθ/r, sinφcosθ/r, -sinθ/r)
"""
function dspher_to_dcart(S, f_φ_div_sinθ, f_θ)
	r = S.r + eps()
   return SVector( - (S.sinφ * f_φ_div_sinθ) + (S.cosφ * S.cosθ * f_θ),
			            (S.cosφ * f_φ_div_sinθ) + (S.sinφ * S.cosθ * f_θ),
			 			                                 - (   S.sinθ * f_θ) ) / r
end

dspher_to_dcart(r, sinφ, cosφ, sinθ, cosθ, f_φ_div_sinθ, f_θ) = 
   	SVector( - (sinφ * f_φ_div_sinθ) + (cosφ * cosθ * f_θ),
			            (cosφ * f_φ_div_sinθ) + (sinφ * cosθ * f_θ),
			 			                                 - (   sinθ * f_θ) ) / (r+eps(r))

function dspher_to_dcart(S, f_r_times_r, f_φ_div_sinθ, f_θ)
	r = S.r + eps()
    return SVector((S.sinθ * S.cosφ * f_r_times_r) - (S.sinφ * f_φ_div_sinθ) + (S.cosφ * S.cosθ * f_θ),
						(S.sinθ * S.sinφ * f_r_times_r) + (S.cosφ * f_φ_div_sinθ) + (S.sinφ * S.cosθ * f_θ),
								(S.cosθ * f_r_times_r) - (S.sinθ * f_θ))/r
end

dspher_to_dcart(r, sinφ, cosφ, sinθ, cosθ, f_r_times_r, f_φ_div_sinθ, f_θ) = 
   	SVector( (sinθ * cosφ * f_r_times_r) - (sinφ * f_φ_div_sinθ) + (cosφ * cosθ * f_θ),
	   				(sinθ * sinφ * f_r_times_r) + (cosφ * f_φ_div_sinθ) + (sinφ * cosθ * f_θ),
					   (cosθ * f_r_times_r) - (sinθ * f_θ)) / (r+eps(r)) 


include("alp.jl")

include("cylm.jl")

include("crlm.jl")

include("rylm.jl")

include("rrlm.jl")

# ---------------------------- Auxiliary functions 

function rand_sphere() 
	r = @SVector randn(3)
	return r / norm(r)
end