using chembases
using Test

@testset "chembases.jl" begin 
    @testset "primitive_GTO" begin include("test_primitive_GTO.jl"); end
    @testset "basis" begin include("test_basis.jl"); end
end
