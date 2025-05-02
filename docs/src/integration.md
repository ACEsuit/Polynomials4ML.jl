
# ChainRules

We aim to provide `ChainRules.jl` integration for all model components. At present, we have focused on providing the `rrule` interface. If `basis` is a polynomial basis then one can obtain its value and pullback via the `ChainRules.jl` interface, 
```julia 
B, pb = rrule(evaluate, basis, X)
B, dP, pb = rrule(evaluate_ed, basis, X)
``` 
Internally, the pullback `pb` will most likely call a custom implementation of the pullback operation.

If any `rrule`s are missing or not working as expected, please file an issue. 

