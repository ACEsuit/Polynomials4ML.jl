
# Public API 

This page documents the public API, i.e. the list of bases and functions that are considered relatively stable and for which we aim to strictly impose semver backward compatibility. The basis sets that are considered stable are the following (please see inline documentation for initialization): 

* Several classes of orthogonal polynomials [`OrthPolyBasis1D3T`](@ref)
   - General Jacobi [`jacobi_basis`](@ref)
   - Legendre [`legendre_basis`](@ref)
   - Chebyshev [`chebyshev_basis`](@ref)
   - Discrete distribution [`orthpolybasis`](@ref) 
* 2D harmonics: 
   - Complex trigonometric polynomials [`CTrigBasis`](@ref)
   - Real trigonometric polynomials [`RTrigBasis`](@ref)
* 3D harmonics: 
   - Complex spherical harmonics [`CYlmBasis`](@ref)
   - Real spherical harmonics [`RYlmBasis`](@ref)
   - Complex solid harmonics [`CRlmBasis`](@ref)
   - Real solid harmonics [`RRlmBasis`](@ref)
* Chebyshev polynomials of the first kind [`ChebBasis`](@ref)
   - this approach computes the basis on the go when it is compiled
   - it does not store the recursion coefficients like what is done in the orthogonal polynomials
* Various quantum chemistry related radial basis functions. (experimental)
   
## In-place Evaluation  

This section documents the in-place evaluation interface. *All* basis sets implemented in this package should provide this interface as a minimal requirement. 

```julia
evaluate!(P, basis, X)
evaluate_ed!(P, dP, basis, X)
evaluate_ed2!(P, dP, ddP, basis, X)
```

* `basis` : an object defining one of the basis sets 
* `X` : a single input or array of inputs. 
* `P` : array containing the basis values 
* `dP` : array containing derivatives of basis w.r.t. inputs 
* `ddP` : array containing second derivatives of basis w.r.t. inputs 

If `X` is a single input then this should normally be a `Number` or a `StaticArray` to distinguish it from collections of inputs. `X` can also be an `AbstractArray` of admissible inputs, e.g., `Vector{<: Number}`. 

If `X` is a single input then `P`, `dP`, `ddP` will be `AbstractVector`. If `X` is an `AbstractVector` of inputs then `P`, `dP`, `ddP` must be `AbstractMatrix`, and so forth. 

The output arrays `P`, `dP`, `ddP` must be sufficiently large in each dimension to accomodate the size of the input and the size of the basis, but the sizes need not match exactly. It is up to the caller to ensure matching array sizes if this is needed.


## Allocating Evaluation

This section documents the allocating evaluation interface. All basis sets should implement this interface.

```julia
P = evaluate(basis, X)
P, dP = evaluate_ed(basis, X)
P, dP, ddP = evaluate_ed2(basis, X)
```
The output types of `P, dP, ddP` are guarnateed to be `AbstractArray`s but may otherwise change between package versions. The exact type should not be relied upon when using this package. 

The meaning of the different symbols is exactly the same as described above. The only difference is that the output containers `P`, `dP`, `ddP` are now allocated. 
Their type should be stable (if not, please file a bug report), but unspecified in the sense that the output type is not semver-stable for the time being. 
If you need a sem-ver stable output then it is best to follow the above with a `collect`.

