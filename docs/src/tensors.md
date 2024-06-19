
## API for Tensor Layers

P4ML also implements a few standard layers that occur in atomic cluster expansion (ACE) type models. These are all expressed as (usually symmetric) tensor operations. These are documented here. Some of these operations are quite unique (sparse symmetric tensor contractions) while others are more standard (`LinearLayer') and are provided here for conformity with our interface. 

* LinearLayer  : todo add doc string
* Sparse product : todo add doc string
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

### Pullbacks

All tensor layers have custom pullbacks implemented that can be accessed via non-allocating or allocating calls: 
```julia 
pullback!(∂X, ∂P, layer, X)
∂X = pullback(∂P, layer, X)
```

### Pushforwards and Second-order derivatives

Pushforwards and reverse-over-reverse are implemented using ForwardDiff. This is quasi-optimal even for reverse-over-reverse due to the fact that it can be interpreted as a directional derivative on evaluate and pullback (after swapping derivatives). As a matter of fact, we generally recommend to not use these directly. ChainRules integration would give an easier use-pattern. For optimal performance the same technique to an entire model architecture rather than to each individual layer. This would avoid several unnecessary intermediate allocations.

The syntax for pushforwards is straightforward:
```julia
pushforward!(P, ∂P, layer, X, ∂X)
P, ∂P = pushforward(layer, X, ∂X)
```

For second-order pullbacks the syntax is 
```julia
∇_∂P, ∇_X = pullback2(∂∂X, ∂P, layer, X)
pullback2!(∇_∂P, ∇_X, ∂∂X, ∂P, layer, X)
```

### Bumper and WithAlloc usage

Using the `WithAlloc.jl` interface these can be used conveniently as follows (always from within a `@no_escape` block)
```julia 
∂X = @withalloc pullback!(∂P, layer, X)
P, ∂P = @withalloc pushforward!(layer, X, ∂X)
```
