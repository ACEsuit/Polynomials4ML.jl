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

import LuxCore: AbstractLuxLayer, initialparameters, initialstates                 

using Random: AbstractRNG   


function _generate_input end 
function _generate_batch end 

# TODO: unclear that natural_indices and index do. Is this specified properly? 
#       it seems that natural_indices shouls give the "natural" indices for the 
#       basis functions in the order that they are stored. But e.g. this 
#       is not true for CTrigBasis; should make a decision and document this 
#       properly.
function natural_indices end   # could rename this get_spec or similar ... 
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
# include("orthopolybasis.jl")
# include("discreteweights.jl")
# include("jacobiweights.jl")
include("monomials.jl")
include("chebbasis.jl")

# 2d harmonics / trigonometric polynomials 
include("trig.jl")
include("rtrig.jl")

# 3d harmonics 
include("sphericart.jl")

# quantum chemistry 
# TODO: RESTRUCTURE OR MOVE?
# include("atomicorbitalsradials/atomicorbitalsradials.jl")

# generating product bases (generalisation of tensor products)
# RETIRE - to be discussed
# include("sparseproduct.jl")

# LinearLayer implementation
# this is needed to better play with cached arrays + to give the correct 
# behaviour when the feature dimension is different from expected. 
# RETIRE 
# include("linear.jl")

# generic machinery for wrapping poly4ml bases into lux layers 
# include("lux.jl")

# some nice utility functions to generate basis sets and other things  
include("utils/utils.jl")

# submodule with some useful utilities for writing unit tests 
include("testing.jl")

end
