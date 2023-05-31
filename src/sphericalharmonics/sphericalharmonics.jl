using StaticArrays, LinearAlgebra, LoopVectorization
export CYlmBasis, RYlmBasis, CRlmBasis, RRlmBasis 


# --------------------------------------------------------
#     Coordinates
# --------------------------------------------------------

# SphericalCoords type is defined in `interface.jl`

spher2cart(S::SphericalCoords) = S.r * SVector(S.cosφ*S.sinθ, S.sinφ*S.sinθ, S.cosθ)

function cart2spher(R::AbstractVector) # ; SH = true)
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
	r = S.r
   return SVector( - (S.sinφ * f_φ_div_sinθ) + (S.cosφ * S.cosθ * f_θ),
			            (S.cosφ * f_φ_div_sinθ) + (S.sinφ * S.cosθ * f_θ),
			 			                                 - (   S.sinθ * f_θ) ) / (r+eps(r))
end

# this looks like a leftover function from an experiment to try 
# make a hot inner loop faster by avoiding the SphericalCoords type. 
dspher_to_dcart(r, sinφ, cosφ, sinθ, cosθ, f_φ_div_sinθ, f_θ) = 
   	SVector( - (sinφ * f_φ_div_sinθ) + (cosφ * cosθ * f_θ),
			            (cosφ * f_φ_div_sinθ) + (sinφ * cosθ * f_θ),
			 			                                 - (   sinθ * f_θ) ) / (r+eps(r))

function dspher_to_dcart(S, f_r_times_r, f_φ_div_sinθ, f_θ) # ; SH = true)
	r = S.r
	return SVector((S.sinθ * S.cosφ * f_r_times_r) - (S.sinφ * f_φ_div_sinθ) + (S.cosφ * S.cosθ * f_θ),
					(S.sinθ * S.sinφ * f_r_times_r) + (S.cosφ * f_φ_div_sinθ) + (S.sinφ * S.cosθ * f_θ),
							(S.cosθ * f_r_times_r) - (S.sinθ * f_θ))/ (r+eps(r))
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

export lm2idx, idx2lm, idx2l, maxL 


natural_indices(basis::XlmBasis) = 
		[ NamedTuple{(:l, :m)}(idx2lm(i)) for i = 1:length(basis) ]

degree(basis::XlmBasis, b::NamedTuple) = b.l 


import Base.==
==(B1::XlmBasis, B2::XlmBasis) =
		(B1.alp == B2.alp) && (typeof(B1) == typeof(B2))


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


function cart2spher(basis::XlmBasis, X::AbstractVector{<: AbstractVector})
	ST = SphericalCoords{eltype(eltype(X))}
	S = acquire!(basis.tmp, :S, (length(X),), ST)
	for i = 1:length(X) 
		S[i] = cart2spher(X[i])
	end
	return S 
end

cart2spher(basis::XlmBasis, x::AbstractVector{<: Real}) = cart2spher(x) 


_acqu_alp!(sym::Symbol, basis::XlmBasis, S::SphericalCoords) = 
		acquire!(basis.tmp, sym, (length(basis.alp),), _valtype(basis.alp, S))

_acqu_alp!(sym::Symbol, basis::XlmBasis, S::AbstractVector{<: SphericalCoords}) = 
		acquire!(basis.tmp, sym, (length(S), length(basis.alp)), _valtype(basis.alp, eltype(S)))

_acqu_P!(  basis::XlmBasis, S) = _acqu_alp!(:alpP,   basis, S)
_acqu_dP!( basis::XlmBasis, S) = _acqu_alp!(:alpdP,  basis, S)
_acqu_ddP!(basis::XlmBasis, S) = _acqu_alp!(:alpddP, basis, S)


# ---------------------------- Auxiliary functions 

function rand_sphere() 
	r = @SVector randn(3)
	return r / norm(r)
end