using Polynomials4ML
using Test

@testset "Polynomials4ML.jl" begin 

    # 1D Polynomials 
    @testset "OrthPolyBasis1D3T" begin include("test_op1d3t.jl"); end
    @testset "DiscreteWeights" begin include("test_discreteweights.jl"); end

    # 2D Harmonics 
    @testset "TrigonometricPolynomials" begin include("test_trig.jl"); end
    @testset "Real Trig Polys" begin include("test_rtrig.jl"); end

    # 3D Harmonics 
    @testset "Complex SphericalHarmonics" begin include("test_cylm.jl"); end
    @testset "Real Spherical Harmonics" begin include("test_rylm.jl"); end
    @testset "Complex Solid Harmonics" begin include("test_crlm.jl"); end
    @testset "Real Solid Harmonics" begin include("test_rrlm.jl"); end

    # Quantum Chemistry 
    @testset "Rnl" begin include("test_rnl.jl"); end

    # Misc
    @testset "Flex Array Interface" begin include("test_flex.jl"); end 
    @testset "Sparse Product" begin include("test_sparseproduct.jl"); end 
    @testset "Lux" begin include("test_lux.jl"); end 
end
