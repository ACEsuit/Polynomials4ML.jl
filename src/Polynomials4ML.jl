module Polynomials4ML

# -------------- Import Bumper and related things ---------------

using Bumper, WithAlloc, StrideArrays
import WithAlloc: whatalloc 

# -------------- import ACEbase stuff 
# not so clear this is still needed? 

import ACEbase
import ACEbase: evaluate, evaluate_d, evaluate_ed, evaluate_dd, evaluate_ed2, 
                evaluate!, evaluate_d!, evaluate_ed!, evaluate_ed2!
import ACEbase.FIO: read_dict, write_dict

using LuxCore, Random
import ChainRulesCore: rrule, frule, NoTangent, ZeroTangent

import LuxCore: AbstractExplicitLayer, AbstractExplicitContainerLayer, 
                 initialparameters, initialstates                 

using Random: AbstractRNG     


function natural_indices end   # could rename this get_spec or similar ... 
function index end
function orthpolybasis end
function degree end 

function pullback_evaluate end
function pullback_evaluate! end
function pushforward_evaluate end
function pushforward_evaluate! end

# some stuff to allow bases to overload some lux functionality ... 
# how much of this should go into ACEbase? 
function lux end 

export natural_indices, 
       index, 
       evaluate, 
       evaluate_d, 
       evaluate_dd, 
       evaluate_ed, 
       evaluate_ed2, 
       evaluate!, 
       evaluate_ed!, 
       evaluate_ed2!, 
       orthpolybasis, 
       degree, 
       pullback_evaluate!, 
       pushforward_evaluate!, 
       pullback_evaluate, 
       pushforward_evaluate



# generic fallbacks for a lot of wrapper kind of functionality 
include("interface.jl")

# static product - used throughout several layers
include("staticprod.jl")

# polynomials 
include("orthopolybasis.jl")
include("discreteweights.jl")
include("jacobiweights.jl")
include("monomials.jl")
include("chebbasis.jl")

# 2d harmonics / trigonometric polynomials 
include("trig.jl")
include("rtrig.jl")

# 3d harmonics 
# include("sphericalharmonics/sphericalharmonics.jl")

# quantum chemistry 
include("atomicorbitalsradials/atomicorbitalsradials.jl")

# generating product bases (generalisation of tensor products)
include("sparseproduct.jl")

# LinearLayer implementation
# this is needed to better play with cached arrays + to give the correct 
# behaviour when the feature dimension is different from expected. 
# include("linear.jl")

# generic machinery for wrapping poly4ml bases into lux layers 
include("lux.jl")

# basis components to implement cluster expansion methods
include("ace/sparseprodpool.jl")
include("ace/symmprod_dag.jl")
include("ace/symmprod_dag_kernels.jl")
include("ace/simpleprodbasis.jl")
include("ace/sparsesymmprod.jl")

# some nice utility functions to generate basis sets and other things  
include("utils/utils.jl")

# submodule with some useful utilities for writing unit tests 
include("testing.jl")

end
