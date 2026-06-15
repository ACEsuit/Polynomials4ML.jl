module Polynomials4ML

# -------------- import ACEbase, Bumper, WithAlloc, Lux and related

import ACEbase 

import ACEbase: evaluate, evaluate_d, evaluate_ed,
                evaluate!, evaluate_d!, evaluate_ed!,
                pullback, pullback!, pushforward, pushforward!,
                natural_indices

import ChainRulesCore: rrule, frule, NoTangent, ZeroTangent
import LuxCore: AbstractLuxLayer, initialparameters, initialstates                 

using Bumper, WithAlloc
import WithAlloc: whatalloc 

using KernelAbstractions, GPUArraysCore

using LuxCore, Random, StaticArrays, ChainRulesCore
using ForwardDiff: Dual, extract_derivative
using StaticArrays


using Random: AbstractRNG   

"""
`abstract type AbstractP4MLBasis end`

Annotates types that map a low-dimensional input, scalar or `SVector`,
to a vector of scalars (feature vector, embedding, basis...). Can be used 
as a `Lux` layer. 
"""
abstract type AbstractP4MLBasis <: AbstractLuxLayer end


"""
   _generate_input(basis)

Returns a single randomly generated valid input for `basis`. 
"""
function _generate_input end 

_generate_batch(basis::AbstractP4MLBasis; nbatch = rand(7:16)) = 
         [ _generate_input(basis) for _ = 1:nbatch ]


# `natural_indices(basis)` (owned by ACEbase, imported above) returns a vector
# of "natural" descriptions of the basis functions, in storage order. For
# Chebyshev this is `0:N`; for spherical harmonics it is a vector of `(l, m)`.

"""
   index(basis, k) -> Integer

Given a "natural description" of a basis element return the index of that basis 
element in the computed vector of basis function values. For example, for 
Chebyshev polynomials, `index(basis, n)` returns `n+1`. 
"""
function index end

function orthpolybasis end

export orthpolybasis

# generic fallbacks for a lot of wrapper kind of functionality 
include("interface.jl")
include("generic_ad.jl")

# static product - used throughout several layers
include("staticprod.jl")

# transformed basis 
include("transformed.jl")

# utility function to interpret a lux layer as a P4ML basis 
include("wrappedbasis.jl")
include("withstate.jl")

# polynomials 
include("orthopolybasis.jl")
include("discreteweights.jl")
include("jacobiweights.jl")
include("monomials.jl")
include("chebbasis.jl")
include("bernstein.jl")

# splines 
include("splinify.jl")

# 2d harmonics / trigonometric polynomials 
include("ctrig.jl")
include("rtrig.jl")

# 3d spherical harmonics (real + complex solid/spherical) are owned by SpheriCart
# (with an ACEbase extension for the evaluate interface). The quantum-chemistry
# atomic-orbital basis `AtomicOrbitals = Pn * Dn * Ylm` lives here; the
# SpheriCart-specific glue (complex value types, default-Ylm constructors) is in
# ext/SpheriCartExt.jl, loaded when SpheriCart is available.
include("atomicorbitals/aorbasis.jl")
export AtomicOrbitals, RadialDecay, GaussianDecay, SlaterDecay

# generating product bases (generalisation of tensor products)
# RETIRE - to be discussed?
# include("sparseproduct.jl")

# LinearLayer implementation
# this is needed to better play with cached arrays + to give the correct 
# behaviour when the feature dimension is different from expected. 
# RETIRE - to be discussed? 
# include("linear.jl")

# some nice utility functions to generate basis sets and other things  
include("utils/utils.jl")

# submodule with some useful utilities for writing unit tests 
include("testing.jl")

end
