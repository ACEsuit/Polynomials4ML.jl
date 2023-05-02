
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

The meaning of the different symbols is exactly the same as described above. The only difference is that the output arrays `P`, `dP`, `ddP` are now allocated and will have precise the correct shape to match the shape of the input. 


## Backward Differentiation

The backward differentiation interface is experimental and should not be considered part of the public API. 

[TODO]

## Laplacian 

The laplacian interface is experimental and should not be considered part of the public API. 

For some applications it is important to have a fast evaluation of the laplace operator, which can often be achieved at far lower computational cost than a hessian. For example, spherical harmonics are eigenfunctions of the laplacian while solid harmonics have zero-laplacian. To exploit this we provide both in-place and allocating interfaces to evaluate the laplacians. In addition we provide an interface to evaluate the basis, its gradients as well as the laplacian, analogous to `evaluate_ed2` above. This interface is convenient to evaluate laplacians of chains.

```julia
laplacian!(ΔY, basis, X)
ΔY = laplacian(basis, X)
eval_grad_laplace!(Y, dY, ΔY, basis, X)
Y, dY, ΔY = eval_grad_laplace(basis, X)
```


## Lux  

For now, the Lux interface is experimental and should not be considered part of the public API. 

