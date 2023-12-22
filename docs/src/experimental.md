
# Experimental API 

The interfaces specified below are experimental and not part of the public API yet. Some of it is not even implemented yet and are just being sketched out in separate branches. There is no guarantee that these are provided for all of the exported basis sets, and there is no guarantee of semver-compatible backward compatibility at this point.

## Some General Experimental / Undocumented Objects

* [`SparseProduct`](@ref) : a model layer to form tensor products of features, e.g., tensor product polynomial bases. 

## Laplacian 

The laplacian interface is experimental and should not be considered part of the public API. 

For some applications it is important to have a fast evaluation of the laplace operator, which can often be achieved at far lower computational cost than a hessian. For example, spherical harmonics are eigenfunctions of the laplacian while solid harmonics have zero-laplacian. To exploit this we provide both in-place and allocating interfaces to evaluate the laplacians. In addition we provide an interface to evaluate the basis, its gradients as well as the laplacian, analogous to `evaluate_ed2` above. This interface is convenient to evaluate laplacians of chains.

```julia
laplacian!(ΔY, basis, X)
ΔY = laplacian(basis, X)
eval_grad_laplace!(Y, dY, ΔY, basis, X)
Y, dY, ΔY = eval_grad_laplace(basis, X)
```

## Explicit Backward Differentiation

We implement custom pullbacks for most bases. These  take the form
```julia
∂X = pb_evaluate(basis, ∂B, X, args...)
pb_evaluate!(∂X, basis, ∂B, X, args...)
```
and analogously for the `evaluate_***` variants. The `args...` can differ between different basis sets e.g. may rely on intermediate results in the evaluation of the basis. The `rrule` implementations are wrappers for these.

## Explicit Forward Mode Differentiation

We have started to implement custom pushforwards. These take the form
```julia
B, ∂B = pfwd_evaluate(basis, X, ΔX)
pfwd_evaluate!(B, ∂B, basis, X, ΔX)
```
and analogously for other functions. There are currently no `frule` wrappers, but we plan to provide these in due course. 
