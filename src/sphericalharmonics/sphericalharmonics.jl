

using StaticArrays, LinearAlgebra

export CYlmBasis, RYlmBasis


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

# --------------------------------------------------------


include("alp.jl")

include("cylm.jl")

include("rylm.jl")


const YlmBasis = Union{RYlmBasis, CYlmBasis}


# --------------------------------------------------------
# Indexing 

export lm2idx, idx2lm, idx2l


natural_indices(basis::YlmBasis) = 
		[ NamedTuple{(:l, :m)}(idx2lm(i)) for i = 1:length(basis) ]

degree(basis::YlmBasis, b::NamedTuple) = b.l 


"""
`sizeY(maxL):`
Return the size of the set of spherical harmonics ``Y_{l,m}(θ,φ)`` of
degree less than or equal to the given maximum degree `maxL`
"""
sizeY(maxL) = (maxL + 1) * (maxL + 1)

"""
`lm2idx(l,m):`
Return the index into a flat array of real spherical harmonics `Y_lm`
for the given indices `(l,m)`. `Y_lm` are stored in l-major order i.e.
```
	[Y(0,0), Y(1,-1), Y(1,0), Y(1,1), Y(2,-2), ...]
```
"""
lm2idx(l::Integer, m::Integer) = m + l + (l*l) + 1

index_y(l::Integer, m::Integer)  = lm2idx(l, m)

"""
Inverse of `lm2idx`: given an index into a vector of Ylm values, return the 
`l, m` indices.
"""
function idx2lm(i::Integer) 
	l = floor(Int, sqrt(i-1) + 1e-10)
	m = i - (l + (l*l) + 1)
	return l, m 
end 

"""
Partial inverse of `lm2idx`: given an index into a vector of Ylm values, return the 
`l` index. 
"""
idx2l(i::Integer) = floor(Int, sqrt(i-1) + 1e-10)




# ---------------------------- Auxiliary functions 

function rand_sphere() 
	r = @SVector randn(3)
	return r / norm(r)
end