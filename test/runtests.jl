using Polynomials4ML
using Test

@testset "Polynomials4ML.jl" begin 
    @testset "OrthPolyBasis1D3T" begin include("test_op1d3t.jl"); end
    @testset "DiscreteWeights" begin include("test_discreteweights.jl"); end
end