using Polynomials4ML, LinearAlgebra, StaticArrays, Test, Random, 
      LuxCore 
using Polynomials4ML: evaluate, evaluate_ed
using Polynomials4ML.Testing: print_tf, println_slim 
using ForwardDiff

import Polynomials4ML as P4ML
rng = Random.default_rng()

@info("Testing TransformedBasis")

##

@info("   Scalar input, no transforms")
basis = ChebBasis(5)
transin = nothing
transout = nothing 
tbasis = P4ML.lux(P4ML.TransformedBasis(transin, basis, transout, length(basis)))
X = [ P4ML._generate_input(tbasis.basis) for _ in 1:10 ]
ps, st = LuxCore.setup(rng, tbasis)

P1, _ = tbasis(X, ps, st)
P2 = evaluate(tbasis.basis.basis, X)
println_slim(@test P1 ≈ P2) 

##

@info("   Scalar input, input transform")
basis = ChebBasis(5)
transin = x -> 1 / (1+x)
transout = nothing 
tbasis = P4ML.lux(P4ML.TransformedBasis(transin, basis, transout, length(basis)))
P4ML._generate_input(::typeof(tbasis.basis)) = rand() 
X = [ P4ML._generate_input(tbasis.basis) for _ in 1:10 ]
ps, st = LuxCore.setup(rng, tbasis)

P1, _ = tbasis(X, ps, st)
P2 = evaluate(tbasis.basis.basis, transin.(X))
println_slim(@test P1 ≈ P2) 

##

# @info("   Scalar input, input & output transform")
# basis = ChebBasis(5)
# transin = x -> 1 / (1+x)
# transout =  
# tbasis = P4ML.lux(P4ML.TransformedBasis(transin, basis, transout, length(basis)))
# P4ML._generate_input(::typeof(tbasis.basis)) = rand() 
# X = [ P4ML._generate_input(tbasis.basis) for _ in 1:10 ]
# ps, st = LuxCore.setup(rng, tbasis)

# P1, _ = tbasis(X, ps, st)
# P2 = evaluate(tbasis.basis.basis, transin.(X))
# println_slim(@test P1 ≈ P2) 
