using OrthPolys4ML
using Test

@testset "OrthPolys4ML.jl" begin
    @testset "OrthPolyBasis1D3T" begin include("test_op1d3t.jl"); end 
end
