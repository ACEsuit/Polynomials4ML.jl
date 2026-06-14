using AtomicOrbitals
using Test

@testset "AtomicOrbitals.jl" begin
    @testset "Atomic Orbitals (radial decay)" begin include("test_atorbrad.jl"); end
end
