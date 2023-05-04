
import LuxCore 
import LuxCore: initialparameters, initialstates, 


struct PolyLuxLayer{TB}
   basis::B
   meta::Dict{String, Any}
end



"""
lux(basis) : convert a basis into a lux layer. This assumes that the basis 
accepts an input and produces and out that is a vector. 
"""
function lux(basis::AbstractPoly4MLBasis; 
             isconst=true,)
   if isconst
      return ConstPolyLuxLayer(basis)
   end
   error("for now the basis can't have trainable parameters")
end



# -------------------- ConstPolyLuxLayer --------------------

struct ConstPolyLuxLayer{TB} <:
   basis::B
   meta::Dict{String, Any}
end

ConstPolyLuxLayer(basis) = ConstPolyLuxLayer(basis, Dict{String, Any}())

Base.length(l::ConstPolyLuxLayer) = length(l.basis)

initialparameters(rng::AbstractRNG, l::ConstL) = NamedTuple()

initialstates(rng::AbstractRNG, l::ConstL) = NamedTuple()


function (l::ConstPolyLuxLayer)(x, ps, st)
   return ignore_derivatives() do 
      evaluate(l.basis, x)
   end 
end






