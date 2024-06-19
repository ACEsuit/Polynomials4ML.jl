
## ChainRules and Lux Integration

### ChainRules.jl integration 

We aim to provide `ChainRules.jl` integration for all model components. At present, we have focused on providing the `rrule` interface. If `layer` is a polynomial basis or tensor layer then one can obtain its value and pullback via the `ChainRules.jl` interface, 
```julia 
B, pb = rrule(evaluate, basis, X)
``` 
Internally, the pullback `pb` will most likely call a custom implementation of the pullback operation. Where possible we also implement pullbacks over pullbacks to enable second-order backward differentiation. This can, e.g., be accessed via 
```julia
∂X, pb2 = rrule(pullback, ∂B, basis, X)
```
This is needed e.g. when minimizing a loss function that involves a model derivative.

If any `rrule`s are missing or not working as expected, please file an issue. 

### Lux.jl Integration

Although all bases and tensor layers that we implement here can be used "as is", we also aim to provide wrappers that turn them into `Lux.jl` layers. For any model component `p4ml_layer`, one can simply call 
```julia
lux_layer = lux(p4ml_layer)
```
The resulting object `lux_layer` can then be used to construct networks using the `Lux.jl` package. 

This functionality has so far not been tested extensively and we are again interested get feedback and bug reports.