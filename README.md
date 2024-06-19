# Polynomials4ML.jl

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ACEsuit.github.io/Polynomials4ML.jl/stable/) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ACEsuit.github.io/Polynomials4ML.jl/dev/)
[![Build Status](https://github.com/ACEsuit/Polynomials4ML.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ACEsuit/Polynomials4ML.jl/actions/workflows/CI.yml?query=branch%3Amain)

This package implements a few polynomial basis types, tensor layers, convenient methods for evaluation and derivatives up to second order and (hopefully fast) batched evaluation, for building small and fast ML type models. Layers currently implemented include: 
* Various orthogonal polynomials via 3-point recursion
* Trigonometric polynomials 
* Complex and real spherical and solid harmonics 
* Some quantum chemistry basis sets 
* Utilities to recombine them into (tensor) product basis sets 
* Utilities to implement cluster expansion models

We also aim to provide full `Lux.jl` integration to build layered models. A possible application of this might be to implement various flavours of equivariant neural networks and related models. 
