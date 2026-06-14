using Test
using StaticArrays
using LinearAlgebra
using LuxCore
import Polynomials4ML as P4ML
import AtomicOrbitals
import SpheriCart, ACEbase   # ensure the SpheriCart ACEbase extension is active
using Polynomials4ML: evaluate, evaluate_ed


@testset "GaussianBasis + HyperDual matches evaluate/evaluate_ed" begin
    basis = AtomicOrbitals._rand_gaussian_basis()
    P4ML.Testing.test_hyperdual_consistency(basis)
    P4ML.Testing.test_evaluate_xx(basis)
    P4ML.Testing.test_chainrules(basis)
    P4ML.Testing.test_withalloc(basis; allowed_allocs = 0, single=false)
end

@testset "SlaterBasis + HyperDual matches evaluate/evaluate_ed" begin
    basis = AtomicOrbitals._rand_slater_basis()
    P4ML.Testing.test_hyperdual_consistency(basis)
    P4ML.Testing.test_evaluate_xx(basis)
    P4ML.Testing.test_chainrules(basis)
    P4ML.Testing.test_withalloc(basis; allowed_allocs = 0, single=false)
end

@testset "STOBasis + HyperDual matches evaluate/evaluate_ed" begin
    basis = AtomicOrbitals._rand_sto_basis()
    P4ML.Testing.test_hyperdual_consistency(basis)
    P4ML.Testing.test_evaluate_xx(basis)
    P4ML.Testing.test_chainrules(basis)
    P4ML.Testing.test_withalloc(basis; allowed_allocs = 0, single=false)
end
