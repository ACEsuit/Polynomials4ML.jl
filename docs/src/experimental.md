
# Experimental API 

The interfaces specified below are experimental and not part of the public API yet. There is no guarantee that these are provided for all of the exported basis sets, and there is no guarantee of semver-compatible backward compatibility at this point.

## Backward Differentiation w.r.t. Inputs `X`

We implement "manual" pullbacks w.r.t. the `X` variable only! These  take the form
```julia
∂X = pb_evaluate(basis, X, ∂B, args...)
pb_evaluate!(∂X, basis, X, ∂B, args...)
```
and analogously for the `evaluate_***` variants. The `args...` can differ between different basis sets e.g. may rely on intermediate results in the evaluation of the basis. 

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

[TODO]