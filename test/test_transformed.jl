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
trans = nothing
tbasis = P4ML.TransformedBasis(trans, basis)
X = [ P4ML._generate_input(tbasis) for _ in 1:10 ]
ps, st = LuxCore.setup(rng, tbasis)

P1, _ = tbasis(X, ps, st)
P2 = evaluate(tbasis.basis, X)
println_slim(@test P1 ≈ P2) 

##

@info("   Scalar input, input transform")
basis = ChebBasis(5)
trans = x -> 1 / (1+x)
tbasis = P4ML.TransformedBasis(trans, basis)
P4ML._generate_input(::typeof(tbasis)) = rand() 
X = [ P4ML._generate_input(tbasis) for _ in 1:10 ]
ps, st = LuxCore.setup(rng, tbasis)

P1, _ = tbasis(X, ps, st)
P2 = evaluate(tbasis.basis, trans.(X))
println_slim(@test P1 ≈ P2) 

##

@info("   Vector input transformed to scalar")
basis = ChebBasis(5)
trans = x -> 1 / (1+ sum(x .* x))
tbasis = P4ML.TransformedBasis(trans, basis)
P4ML._generate_input(::typeof(tbasis)) = rand(SVector{3, Float64}) 
X = [ P4ML._generate_input(tbasis) for _ in 1:10 ]
ps, st = LuxCore.setup(rng, tbasis)

P1, _ = tbasis(X, ps, st)
P2 = evaluate(tbasis.basis, trans.(X))
println_slim(@test P1 ≈ P2) 

##

@info("   Vector input transformed to vector")
basis = real_solidharmonics(5)
trans = x -> x / norm(x) 
tbasis = P4ML.TransformedBasis(trans, basis)
P4ML._generate_input(::typeof(tbasis)) = rand(SVector{3, Float64}) 
X = [ P4ML._generate_input(tbasis) for _ in 1:10 ]
ps, st = LuxCore.setup(rng, tbasis)

P1, _ = tbasis(X, ps, st)
P2 = evaluate(tbasis.basis, trans.(X))
println_slim(@test P1 ≈ P2) 
