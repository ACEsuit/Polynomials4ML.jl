using Polynomials4ML
using Test

##
@testset "Polynomials4ML.jl" begin 

    # 1D Polynomials 
    @testset "OrthPolyBasis1D3T" begin include("test_op1d3t.jl"); end
    @testset "DiscreteWeights" begin include("test_discreteweights.jl"); end
    @testset "Chebyshev" begin include("test_cheb.jl"); end 
    @testset "Monomials" begin include("test_mono.jl"); end 

    # 2D & 3D Harmonics 
    @testset "Complex Trigonometric" begin include("test_ctrig.jl"); end
    @testset "Real Trigonometric" begin include("test_rtrig.jl"); end
    @testset "SpheriCart Interface" begin include("test_sphericart.jl"); end

    # Quantum Chemistry 
    @testset "Atomic Orbitals" begin include("test_atorbrad.jl"); end
    
    # Misc
    @testset "Static Prod" begin include("test_staticprod.jl"); end

    # Transformations 
    @testset "Transformed Basis" begin include("test_transformed.jl"); end

    # Test lux interface 
    # @testset "Lux" begin include("test_lux.jl"); end 

    # TODO: restructure or move?? 
    # @testset "Sparse Product" begin include("test_sparseproduct.jl"); end 
    # @testset "Linear layer" begin include("test_linear.jl"); end 
end
