# Polynomials4ML.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ACEsuit.github.io/Polynomials4ML.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ACEsuit.github.io/Polynomials4ML.jl/dev/)
[![Build Status](https://github.com/ACEsuit/Polynomials4ML.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ACEsuit/Polynomials4ML.jl/actions/workflows/CI.yml?query=branch%3Amain)

This package implements a few polynomial basis types, and convenience methods for evaluation and derivatives, fast batched evaluation, for building small and fast ML type models. Layers currently implemented include: 
* Various orthogonal polynomials via 3-point recursion
* Trigonometric polynomials 
* Complex and real spherical and solid harmonics 
* A few quantum chemistry (atomic orbitals) basis sets 
* Interpolate a basis onto splines
* Utilities to recombine them into (tensor) product or compressed basis sets 

We also aim to provide full `Lux.jl` integration. A possible application of this might be to implement various flavours of equivariant neural networks and related models. 
