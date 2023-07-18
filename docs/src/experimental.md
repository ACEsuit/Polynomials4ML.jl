
# Experimental API 

The interfaces specified below are experimental and not part of the public API yet. Some of it is not even implemented yet and are just being sketched out in separate branches. There is no guarantee that these are provided for all of the exported basis sets, and there is no guarantee of semver-compatible backward compatibility at this point.

## Laplacian 

The laplacian interface is experimental and should not be considered part of the public API. 

For some applications it is important to have a fast evaluation of the laplace operator, which can often be achieved at far lower computational cost than a hessian. For example, spherical harmonics are eigenfunctions of the laplacian while solid harmonics have zero-laplacian. To exploit this we provide both in-place and allocating interfaces to evaluate the laplacians. In addition we provide an interface to evaluate the basis, its gradients as well as the laplacian, analogous to `evaluate_ed2` above. This interface is convenient to evaluate laplacians of chains.

```julia
laplacian!(ΔY, basis, X)
ΔY = laplacian(basis, X)
eval_grad_laplace!(Y, dY, ΔY, basis, X)
Y, dY, ΔY = eval_grad_laplace(basis, X)
```


## (Atomic) Cluster Expansion 

Two key operations are probided that are needed for the implementation of the (atomic) cluster expansion. The precise deficitions and interface may still change, so those are also still labelled experimental. 
* [`PooledSparseProduct`](@ref) : implements a merged product basis and pooling operation; in the atomic cluster expansion this is called the atomic basis; in GAP it is called the density projection.
* [`SparseSymmProd`](@ref) : implements a sparse symmetric rank-1 tensor product, in ACE this is called the product basis, in GAP the n-correlations.

Both of those operations have pullbacks implemented, but not `evaluate_ed!` or `evaluate_ed2!`.

## Backward Differentiation w.r.t. Inputs `X`

[WORK IN PROGRESS] We implement "manual" pullbacks w.r.t. the `X` variable. These  take the form
```julia
∂X = pb_evaluate(basis, ∂B, X, args...)
pb_evaluate!(∂X, basis, ∂B, X, args...)
```
and analogously for the `evaluate_***` variants. The `args...` can differ between different basis sets e.g. may rely on intermediate results in the evaluation of the basis. 


## Lux  

[TODO] describe the lux layer interface as it evolves ... 