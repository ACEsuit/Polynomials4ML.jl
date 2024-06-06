# IMPORT 
# some of this functionality is probably duplicated here and can likely  
# be replaced with the generic one in lux.jl 
# for now keep those imports until we properly re-organize the lux 
# implementation

import ChainRulesCore: rrule, NoTangent 

import LuxCore: AbstractExplicitLayer, AbstractExplicitContainerLayer, 
                 initialparameters, initialstates                 

using Random: AbstractRNG                 

# pooled sparse product 
include("sparseprodpool.jl")

# sparse symmetric tensor product 
# include("symmprod_dag.jl")
# include("symmprod_dag_kernels.jl")
include("simpleprodbasis.jl")
include("sparsesymmprod.jl")
