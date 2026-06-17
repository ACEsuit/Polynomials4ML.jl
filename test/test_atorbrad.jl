using Test
using StaticArrays
using LinearAlgebra
using LuxCore
import Polynomials4ML as P4ML
import SpheriCart, ACEbase   # activate the SpheriCart ACEbase + P4ML SpheriCart extensions
using Polynomials4ML: evaluate, evaluate_ed

# NOTE: the `test_withalloc` calls below use `alloc_broken = true`, so the
# zero-allocation assertion is a `@test_broken`. AtomicOrbitals delegates the
# angular basis to SpheriCart, whose KA `compute!` needs a backend-aware output
# array; we therefore evaluate it via SpheriCart's *allocating* API (see
# Polynomials4ML #124), so the eval is no longer allocation-free. Drop
# `alloc_broken` to restore a hard `@test` once SpheriCart can write into a
# Bumper-stack buffer.

@testset "GaussianBasis + HyperDual matches evaluate/evaluate_ed" begin
    basis = P4ML._rand_gaussian_basis()
    P4ML.Testing.test_hyperdual_consistency(basis)
    P4ML.Testing.test_evaluate_xx(basis)
    P4ML.Testing.test_chainrules(basis)
    P4ML.Testing.test_withalloc(basis; allowed_allocs = 0, single=false,
                                alloc_broken=true)
end

@testset "SlaterBasis + HyperDual matches evaluate/evaluate_ed" begin
    basis = P4ML._rand_slater_basis()
    P4ML.Testing.test_hyperdual_consistency(basis)
    P4ML.Testing.test_evaluate_xx(basis)
    P4ML.Testing.test_chainrules(basis)
    P4ML.Testing.test_withalloc(basis; allowed_allocs = 0, single=false,
                                alloc_broken=true)
end

@testset "STOBasis + HyperDual matches evaluate/evaluate_ed" begin
    basis = P4ML._rand_sto_basis()
    P4ML.Testing.test_hyperdual_consistency(basis)
    P4ML.Testing.test_evaluate_xx(basis)
    P4ML.Testing.test_chainrules(basis)
    P4ML.Testing.test_withalloc(basis; allowed_allocs = 0, single=false,
                                alloc_broken=true)
end
