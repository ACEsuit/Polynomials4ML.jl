# --------------------- Jacobi Weights 
# this includes in particular Legendre, Gegenbauere, Chebyshev 

struct JacobiWeights{T}
   α::T
   β::T
   a::T 
   b::T 
end

chebyshev_weights(; a = -1.0, b = 1.0) = JacobiWeights(-0.5, -0.5, a, b)

legendre_weights(; a = -1.0, b = 1.0) = JacobiWeights(0.0, 0.0, a, b)

# function orthpolys(space::WeightedL2{<: JacobiWeights})

# end
