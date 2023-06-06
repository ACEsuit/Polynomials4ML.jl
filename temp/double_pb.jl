using Zygote, ChainRules, LinearAlgebra, ChainRulesCore

import ChainRules: rrule

function f(x, W) 
   return W * sin.(x)
end

function g(y)
   return sum(abs2, y)
end

function gf(x, W) 
   return g(f(x, W))
end


x = rand(10)
W = rand(5, 10)
gf(x, W)

∇gf = (x, W) -> Zygote.gradient(x -> gf(x, W), x)[1]
∇gf(x, W)

L(W) = sum(abs2, ∇gf(x, W))
L(W)

Zygote.gradient(W -> L(W), W)[1]


##

# also works if we add our custom rrules. 

function rrule(::typeof(g), y)
   return g(y), Δ -> (NoTangent(), 2 * Δ * y)
end

function rrule(::typeof(f), x, W)
   y = f(x, W)
   return y, Δ -> (NoTangent(), cos.(x) .* (W' * Δ), Δ * sin.(x)')
end
