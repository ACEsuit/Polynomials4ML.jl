abstract type RlmBasis end

"""
max L degree for which the alp coefficients have been precomputed
"""
maxL(sh::RlmBasis) = sh.alp.L

Base.length(basis::RlmBasis) = sizeY(maxL(basis))


# ---------------------- Indexing

"""
`sizeY(maxL):`
Return the size of the set of spherical harmonics ``Y_{l,m}(θ,φ)`` of
degree less than or equal to the given maximum degree `maxL`
"""
sizeY(maxL) = (maxL + 1) * (maxL + 1)

"""
`index_y(l,m):`
Return the index into a flat array of real spherical harmonics `Y_lm`
for the given indices `(l,m)`. `Y_lm` are stored in l-major order i.e.
```
	[Y(0,0), Y(1,-1), Y(1,0), Y(1,1), Y(2,-2), ...]
```
"""
index_y(l::Integer, m::Integer) = m + l + (l*l) + 1

"""
Inverse of `index_y`: given an index into a vector of Ylm values, return the 
`l, m` indices.
"""
function idx2lm(i::Integer) 
	l = floor(Int, sqrt(i-1) + 1e-10)
	m = i - (l + (l*l) + 1)
	return l, m 
end 

idx2l(i::Integer) = floor(Int, sqrt(i-1) + 1e-10)

# ---------------------- evaluation interface code 
function evaluate(basis::RlmBasis, x::AbstractVector{<: Real})
	Y = acquire!(basis.pool, length(basis), _valtype(basis, x))
	evaluate!(parent(Y), basis, x)
	return Y 
end

function evaluate(basis::RlmBasis, X::AbstractVector{<: AbstractVector})
	Y = acquire!(basis.ppool, (length(X), length(basis)))
	evaluate!(parent(Y), basis, X)
	return Y 
end

