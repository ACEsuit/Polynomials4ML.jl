module Polynomials4ML

include("objectpools.jl")
using Polynomials4ML.ObjectPools: TempArray, CachedArray, ArrayCache


include("interface.jl")

include("orthopolybasis.jl")
include("discreteweights.jl")
include("jacobiweights.jl")

include("trig.jl")
include("sphericalharmonics/sphericalharmonics.jl")

include("testing.jl")

end
