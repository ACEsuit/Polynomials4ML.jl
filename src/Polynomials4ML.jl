module Polynomials4ML

using ObjectPools: ArrayCache, TempArray, acquire!, release!

include("interface.jl")

include("orthopolybasis.jl")
include("discreteweights.jl")
include("jacobiweights.jl")

include("trig.jl")
include("sphericalharmonics/sphericalharmonics.jl")

include("testing.jl")

end
