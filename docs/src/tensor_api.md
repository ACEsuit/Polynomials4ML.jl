
# Tensor Layer API

P4ML also implements a few standard layers that occur in atomic cluster expansion (ACE) models. These are all expressed as (usually symmetric) tensor operations. These are documented here. 

* LinearLayer [`LinearLayer`](@ref)
* Sparse product
* Fused tensor product and pooling [`PooledSparseProduct`](@ref)
* Sparse symmetric product [`SparseSymmProd`](@ref)
* Recursive sparse symmetric product implementation [`SparseSymmProdDAG`](@ref)

Their usage differs slightly from the polynomial embeddings. Evaluating a layer is the same and can be done both in-place and allocating, e.g., 
```julia
abasis::PooledSparseProduct
evaluate!(A, abasis, BB)
A = evaluate(abasis, BB)
```
We refer to the individual documentation for the details of the arguments to each layer.

All tensor layers can be conveniently used in a non-allocating way from a Bumper `@no_escape` block, e.g., 
```julia 
A = @withalloc evaluate!(absis, BB)
```

## Pullbacks and Pushforwards

All tensor layers have hand-written pullbacks implemented via 
```julia 
pullback_evaluate!(∂X, ∂P, layer, X)
∂X = pullback_evaluate(∂P, layer, X)
pushforward_evaluate!(P, ∂P, layer, X, ∂X)
P, ∂P = pushforward_evaluate(layer, X, ∂X)
```
Using the `WithAlloc.jl` interface these can again be used as follows, 
```julia 
∂X = @withalloc pullback_evaluate!(∂P, layer, X)
P, ∂P = @withalloc pushforward_evaluate!(layer, X, ∂X)
```

Second-order pushbacks  ... TODO 
<!--
pb_pb_evaluate!(∂X, ∂P, layer, X)
pb_pb_evaluate!(∂X, ∂P, layer, X) 
-->

## ChainRules integration 


## Lux integration 
