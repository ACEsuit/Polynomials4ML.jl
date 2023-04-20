using Polynomials4ML
using Test

@testset "Polynomials4ML.jl" begin 
    @testset "OrthPolyBasis1D3T" begin include("test_op1d3t.jl"); end
    @testset "DiscreteWeights" begin include("test_discreteweights.jl"); end
    @testset "TrigonometricPolynomials" begin include("test_trig.jl"); end
    @testset "Real Trig Polys" begin include("test_rtrig.jl"); end
    @testset "SphericalHarmonics" begin include("test_cylm.jl"); end
    @testset "SphericalHarmonics" begin include("test_rylm.jl"); end
end