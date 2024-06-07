
# Public API 

This page documents the public API for polynomial bases: the list of bases and functions that are considered relatively stable and for which we aim to strictly impose semver backward compatibility. The basis sets that are considered stable are the following (please see inline documentation for initialization): 

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

## In-place Evaluation  

This section documents the in-place evaluation interface. The polynomial basis sets implemented in this package should provide this interface as a minimal requirement. Because these mappings are all low-to-high dimensional, and are almost never a computational bottleneck, we do not current support pullbacks and pushforwards but only classical evaluation and point-wise differentiation, for either single inputs or batches of inputs. 

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
The output types of `P, dP, ddP` are guaranteed to be `AbstractArray`s but may otherwise change between package versions. The exact type should not be relied upon when using this package.

The meaning of the different symbols is exactly the same as described above. The only difference is that the output containers `P`, `dP`, `ddP` are now allocated. 
Their type should be stable (if not, please file a bug report), but unspecified in the sense that the output type is not semver-stable for the time being. 
If you need a sem-ver stable output then it is best to follow the above with a `collect` or to wrap the non-allocation versions. 

## Using the `WithAlloc.jl` Bumper extension 

The package `WithAlloc` introduces a function `whatalloc` that allows one to specify the output arrays required for an in-place evaluation. It furthermore provides a macro `@withalloc` to wrap this functionality conveniently. For example, 
```julia
@no_escape begin 
   basis = legendre_basis(10) 
   X = 2 * rand(100) .- 1
   P1 = @withalloc evaluate!(basis, X)
   P2, dP2 = @withalloc evaluate_ed!(basis, X)
   ;
end 
```
The arrays `P1, P2, dP2` are Bumper-allocated i.e. are not allowed to leave the no-escape block. Please see `[WithAlloc.jl](https://github.com/ACEsuit/WithAlloc.jl)` and `[Bumper.jl](https://github.com/MasonProtter/Bumper.jl)` for more details. If output arrays are to be used outside of the local scope then the allocating functions `evaluate`, `evaluate_ed` etc, should be used or array allocation managed differently. 

## ChainRules.jl integration 

We aim to provide `ChainRules.jl` integration for all model components. At present, we have focused on providing the `rrule` interface. If `basis` is a polynomial basis (or other model component) then one can obtain its value and pullback via the `ChainRules.jl` interface, 
```julia 
B, pb = rrule(evaluate, basis, X)
``` 
Internally, the pullback `pb` will most likely call a custom implementation of the pullback operation. Where possible we also implement pullbacks over pullbacks to enable second-order backward differentiation. This is needed when minimizing a loss function that involves a model derivative.

If any `rrule`s are missing or not working as expected, please file an issue. 


## Lux.jl Integration

Although all bases and models components that we implement here can be used "as is", we also aim to provide wrappers that turn them into `Lux.jl` layers. For any basis or model component `basis`, one can simply call 
```julia
l_basis = lux(basis)
```
The resulting object `l_basis` can then be used to construct networks using the `Lux.jl` package. 