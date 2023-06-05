# Polynomials4ML.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ACEsuit.github.io/Polynomials4ML.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ACEsuit.github.io/Polynomials4ML.jl/dev/)
[![Build Status](https://github.com/ACEsuit/Polynomials4ML.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ACEsuit/Polynomials4ML.jl/actions/workflows/CI.yml?query=branch%3Amain)

This package implements a few polynomial basis types, convenient methods for evaluation, derivatives up to second order and (hopefull fast) batched evaluation. The bases currently implemented include: 
* `OrthPolyBasis1D3T` : univariate polynomial bases specified in terms of the 3-point recursion. Convenient constructors are provided for Jacobi polynomials (`jacobi_basis`, `legendre_basis`, `chebyshev_basis`) and for orthogonality w.r.t. a discrete weight distribution (cf `DiscreteWeights`)
* `MonoBasis` : monomials, included mostly for completeness
* `CTrigBasis` : complex trigonometric polynomials 
* `RTrigBasis` : real trigonometric polynomials 
* `RTrigBasis` : real trigonometric polynomials 
* `CYlmBasis` : complex spherical harmonics 
* `RYlmBasis` : real spherical  harmonics 
* `CRlmBasis` : complex solid harmonics 
* `RRlmBasis` : real solid harmonics 
* `ChebBasis` : chebyshev polynomials of the first kind
* several radial bases for atomic orbitals: Slater, STO, STO-NG


## Basic Usage 

Generate a basis: by default this generates not the standard Chebyshev, Legendre, etc, but a normalized version i.e. the basis functions will have unit norm in the corresponding weighted L2 norm. 
```julia 
using Polynomials4ML 

# polynomial degree - 1 (length of basis)
N = 15 

# standard orthogonal bases 
cheb = chebyshev_basis(N)
legendre = legendre_basis(N) 
jacobi = jacobi_basis(N, 0.5, 0.75)
trig = CTrigBasis(N)

# orthogonal polynomials with discrete weights 
X = 2*rand(100) .- 1
W = 1 .+ rand(100)
dbasis = orthpolybasis(N, X, W, :normalize)
``` 

Evaluate a basis; take `basis` one of the above
```julia 
basis = cheb
# assume that [0, 1] is part of the domain of the basis
x = rand()

# evaluate the basis
P = basis(x) 
P = evaluate(basis, x)

# basis derivatives 
dP = evaluate_d(basis, x)
P, dP = evaluate_ed(basis, x)

# second derivatives 
ddP = evaluate_dd(basis, x)
P, dP, ddP = evaluate_ed2(basis, x)
```
For more efficient non-allocating in-place evaluation, see the documenation. 
