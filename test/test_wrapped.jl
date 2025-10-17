

using Polynomials4ML, LinearAlgebra, StaticArrays, Test, Random, 
      LuxCore, Lux
using Polynomials4ML: evaluate, evaluate_ed
using Polynomials4ML.Testing: print_tf, println_slim 
using ForwardDiff
using ForwardDiff: Dual 

import Polynomials4ML as P4ML
import ForwardDiff as FD

rng = Random.default_rng()

@info("Testing ChainedBasis")

##

basis = ChebBasis(5)
trans = x -> 1 ./ (1 .+ x)

# old approach 
tbasis = P4ML.TransformedBasis(trans, basis)
P4ML._generate_input(::typeof(tbasis)) = rand() 

ps0, st0 = LuxCore.setup(rng, tbasis)
X = [ P4ML._generate_input(tbasis) for _ in 1:10 ]

x = 0.5
b0 = evaluate(tbasis, x, ps0, st0)
B0 = evaluate(tbasis, X, ps0, st0)

##
# new approach 
wrb1 = P4ML.wrapped_basis(
               Chain(; trans=WrappedFunction(trans), basis=basis), 
               1.0) 
ps1, st1 = LuxCore.setup(rng, wrb1)
b1 = evaluate(wrb1, x, ps1, st1)
B1 = evaluate(wrb1, X, ps1, st1)

b1 ≈ b0
B1 ≈ B0


(b1a, db1a) = evaluate_ed(wrb1, x, ps1, st1)
(B1a, dB1a) = evaluate_ed(wrb1, X, ps1, st1)

b1a ≈ b0
B1a ≈ B0

fw_b1a = ForwardDiff.derivative(x -> wrb1.l(x, ps1, st1)[1], x)
fw_b1a ≈ db1a

##
# a slightly more complicated composed basis 

len_basis = length(basis)
wrb2 = P4ML.wrapped_basis(
                Chain(; trans = WrappedFunction(trans), 
                       basis = basis, 
                       linear = P4ML.Utils.LinL(len_basis, len_basis ÷ 2) ), 
                1.0 )
ps2, st2 = LuxCore.setup(rng, wrb2)

b2 = evaluate(wrb2, X[1], ps2, st2)
b2 ≈ ps2.linear.W * basis(trans(X[1]))

B2 = evaluate(wrb2, X, ps2, st2)
B2[1,:] ≈ b2

b2a, db2a = evaluate_ed(wrb2, X[1], ps2, st2)
B2a, dB2a = evaluate_ed(wrb2, X, ps2, st2)
fw_b2a = FD.derivative(x -> wrb2.l(x, ps2, st2)[1], X[1])
b2a ≈ b2 ≈ B2a[1,:]
db2a ≈ dB2a[1,:] ≈ fw_b2a

##

wrb2_wrap = P4ML.WithState(wrb2, ps2, st2)
evaluate(wrb2_wrap, X[1]) ≈ b2
all(evaluate_ed(wrb2_wrap, X) .≈ (B2a, dB2a))

P4ML._generate_input(::typeof(wrb2_wrap)) = rand()
# P4ML.Testing.test_evaluate_xx(wrb2_wrap)
P4ML.Testing.test_chainrules(wrb2_wrap)

##

P4ML.Testing.test_ka_evaluate(wrb2_wrap)
