using Polynomials4ML
using Test

@testset "Polynomials4ML.jl" begin 

    # 1D Polynomials 
    @testset "OrthPolyBasis1D3T" begin include("test_op1d3t.jl"); end
    @testset "DiscreteWeights" begin include("test_discreteweights.jl"); end
    @testset "Chebyshev" begin include("test_cheb.jl"); end 

    # 2D Harmonics 
    @testset "TrigonometricPolynomials" begin include("test_trig.jl"); end
    @testset "Real Trig Polys" begin include("test_rtrig.jl"); end

    # 3D Harmonics 
    @testset "Complex SphericalHarmonics" begin include("sphericalharmonics/test_cylm.jl"); end
    @testset "Real Spherical Harmonics" begin include("sphericalharmonics/test_rylm.jl"); end
    @testset "Complex Solid Harmonics" begin include("sphericalharmonics/test_crlm.jl"); end
    @testset "Real Solid Harmonics" begin include("sphericalharmonics/test_rrlm.jl"); end

    # Quantum Chemistry 
    @testset "Atomic Orbitals Radials" begin include("test_atorbrad.jl"); end

    # ACE 
    @testset "SparsePooledProduct" begin include("ace/test_sparseprodpool.jl"); end 
    @testset "Sparse Symmetric Product" begin include("ace/test_prodbasis1.jl"); end 

    # Misc
    @testset "Static Prod" begin include("test_staticprod.jl"); end
    @testset "Flex Array Interface" begin include("test_flex.jl"); end 
    @testset "Sparse Product" begin include("test_sparseproduct.jl"); end 
    @testset "Lux" begin include("test_lux.jl"); end 
end
