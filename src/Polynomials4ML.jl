module Polynomials4ML

using ObjectPools: ArrayCache, TempArray, acquire!, release!

import ACEbase
import ACEbase: evaluate, evaluate_d, evaluate_ed, evaluate_d2, evaluate_ed2, 
                evaluate!, evaluate_d!, evaluate_ed!, evaluate_d2!, evaluate_ed2!
import ACEbase.FIO: read_dict, write_dict

function natural_indices end 
function index end
function orthpolybasis end
function degree end 

export natural_indices, 
       index, 
       evaluate, 
       evaluate_d, 
       evaluate_d2, 
       evaluate_ed, 
       evaluate_ed2, 
       evaluate!, 
       evaluate_ed!, 
       evaluate_ed2!, 
       orthpolybasis, 
       degree 



# some interface functions that we should maybe retire 
include("interface.jl")

include("orthopolybasis.jl")
include("discreteweights.jl")
include("jacobiweights.jl")

include("monomials.jl")

include("trig.jl")
include("rtrig.jl")

include("sphericalharmonics/sphericalharmonics.jl")

include("testing.jl")

end
