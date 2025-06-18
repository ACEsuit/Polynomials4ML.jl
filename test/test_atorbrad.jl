using LinearAlgebra, StaticArrays, Test, Printf, Polynomials4ML
using Polynomials4ML: evaluate, evaluate_ed
using Polynomials4ML.Testing: print_tf, println_slim 
using ForwardDiff
using ACEbase.Testing: fdtest

import Polynomials4ML as P4ML

##

@info("Testing GaussianBasis")
basis = P4ML._rand_gaussian_basis()

@info("      correctness of evaluation")
x = P4ML._generate_input(basis)
P = evaluate(basis, x)
P1, dP1 = evaluate_ed(basis, x)

P4ML.Testing.test_evaluate_xx(basis)
P4ML.Testing.test_chainrules(basis)

# Test is broken - reshape is causing this, hence single-input test is turned off
P4ML.Testing.test_withalloc(basis; allowed_allocs = 0, single=false)

##
# these are scripts to replicate and check this allocation problem. 
# strangely it doesn't occur for the other bases. Only for AtomicOrbtials. 
#
# using BenchmarkTools

# P, dP = evaluate_ed(basis, x)
# @btime P4ML.evaluate_ed!($P, $dP, $basis, $x)

# @profview let basis=basis, X=x, P=P, dP=dP 
#     for _ = 1:1_000_000 
#         P4ML.evaluate_ed!(P, dP, basis, X)
#     end
# end

##

@info("Testing SlaterBasis")
basis = P4ML._rand_slater_basis()

@info("      correctness of evaluation")
x = P4ML._generate_input(basis)
P = evaluate(basis, x)
P1, dP1 = evaluate_ed(basis, x)

P4ML.Testing.test_evaluate_xx(basis)
P4ML.Testing.test_chainrules(basis)
P4ML.Testing.test_withalloc(basis; allowed_allocs = 0, single=false)

##

@info("Testing STOBasis")
basis = P4ML._rand_sto_basis()

@info("      correctness of evaluation")
x = P4ML._generate_input(basis)
P = evaluate(basis, x)
P1, dP1 = evaluate_ed(basis, x)


P4ML.Testing.test_evaluate_xx(basis)
P4ML.Testing.test_chainrules(basis)
P4ML.Testing.test_withalloc(basis; allowed_allocs = 0, single=false)
