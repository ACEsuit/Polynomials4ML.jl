module Polynomials4ML

# -------------- import ACEbase, Bumper, WithAlloc, Lux and related

import ACEbase
import ACEbase: evaluate, evaluate_d, evaluate_ed, 
                evaluate!, evaluate_d!, evaluate_ed!

using Bumper, WithAlloc
import WithAlloc: whatalloc 

using KernelAbstractions, GPUArraysCore

using LuxCore, Random, StaticArrays
import ChainRulesCore: rrule, frule, NoTangent, ZeroTangent
using HyperDualNumbers: Hyper
using ForwardDiff: Dual, extract_derivative
using StaticArrays
import LuxCore: AbstractLuxLayer, initialparameters, initialstates                 

using Random: AbstractRNG   

"""
`abstract type AbstractP4MLBasis end`

Annotates types that map a low-dimensional input, scalar or `SVector`,
to a vector of scalars (feature vector, embedding, basis...). 
"""
abstract type AbstractP4MLBasis end


"""
   _generate_input(basis)

Returns a single randomly generated valid input for `basis`. 
"""
function _generate_input end 

_generate_batch(basis::AbstractP4MLBasis; nbatch = rand(7:16)) = 
         [ _generate_input(basis) for _ = 1:nbatch ]


""" 
   natural_indices(basis) -> AbstractVector

Returns an abstract vector of "natural" descriptions of the basis functions in 
the order that they are stored in the computed vector of basis function values.
For example, for Chebyshev polynomials, `natural_indices(basis)` returns 
`0:N`, where `N+1` is the length of the basis. For Spherical Harmmonics, 
a natural description requires two indices `(l, m)`, so the output will be a 
vector of tuples.

At the moment, this function is used only for inspection and testing so no 
strict format is enforced.
"""
function natural_indices end   # could rename this get_spec or similar ... 

"""
   index(basis, k) -> Integer

Given a "natural description" of a basis element return the index of that basis 
element in the computed vector of basis function values. For example, for 
Chebyshev polynomials, `index(basis, n)` returns `n+1`. 
"""
function index end

function orthpolybasis end
function degree end 

function pullback! end
function pullback end
function pushforward end
function pushforward! end

# some stuff to allow bases to overload some lux functionality ... 
# how much of this should go into ACEbase? 
function lux end 

export orthpolybasis

# generic fallbacks for a lot of wrapper kind of functionality 
include("interface.jl")
include("generic_ad.jl")

# static product - used throughout several layers
include("staticprod.jl")

# polynomials 
include("orthopolybasis.jl")
include("discreteweights.jl")
include("jacobiweights.jl")
include("monomials.jl")
include("chebbasis.jl")
include("bernstein.jl")
# 2d harmonics / trigonometric polynomials 
include("ctrig.jl")
include("rtrig.jl")

# 3d harmonics 
include("sphericart.jl")

# quantum chemistry 
include("atomicorbitals/atomicorbitals.jl")

# generating product bases (generalisation of tensor products)
# RETIRE - to be discussed?
# include("sparseproduct.jl")

# LinearLayer implementation
# this is needed to better play with cached arrays + to give the correct 
# behaviour when the feature dimension is different from expected. 
# RETIRE - to be discussed? 
# include("linear.jl")

# generic machinery for wrapping poly4ml bases into lux layers 
include("lux.jl")

# some nice utility functions to generate basis sets and other things  
include("utils/utils.jl")

# submodule with some useful utilities for writing unit tests 
include("testing.jl")

end
