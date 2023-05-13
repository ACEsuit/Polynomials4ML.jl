module ACE

using ObjectPools: FlexArrayCache, FlexArray, 
                   ArrayPool, TSafe, 
                   acquire!, release!

import ChainRulesCore: rrule, NoTangent 

import LuxCore: AbstractExplicitLayer, AbstractExplicitContainerLayer, 
                 initialparameters, initialstates                 

using Random: AbstractRNG                 

include("sparseprod.jl")

include("symmprod_dag.jl")
include("symmprod_dag_kernels.jl")

include("simpleprodbasis.jl")
include("sparsesymmprod.jl")

include("utils/utils.jl")
include("testing.jl")

end
