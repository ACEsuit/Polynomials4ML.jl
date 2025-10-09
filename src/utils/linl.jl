
import LuxCore
import LuxCore: AbstractLuxLayer
using Random: AbstractRNG

"""
   struct LinL

A very basic linear layer that is compatible with the memory layout 
of P4ML.       
"""
struct LinL <: AbstractLuxLayer
   in_dim::Int 
   out_dim::Int 
end

LuxCore.initialparameters(rng::AbstractRNG, l::LinL)  = 
      ( W = randn(rng, l.out_dim, l.in_dim) * sqrt(2 / (l.in_dim + l.out_dim)), )

LuxCore.initialstates(rng::AbstractRNG, l::LinL) = NamedTuple()

(l::LinL)(x::AbstractVector, ps, st) = ps.W * x, st 

(l::LinL)(X::AbstractMatrix, ps, st) = X * transpose(ps.W), st 
