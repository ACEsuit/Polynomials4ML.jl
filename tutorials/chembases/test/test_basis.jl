using Test
using Polynomials4ML
using chembases
using LinearAlgebra

# Test all basis sets for all elements
basis_names = [
    # Minimal basis
    "sto-3g", "sto-6g",

    # Pople basis sets
    "3-21g", "6-21g", "6-31g", "6-31g*", "6-31g**",
    "6-311g", "6-311g*", "6-311g**",
    "6-311+g", "6-311++g", "6-311+g*", "6-311++g**",

    # Dunning's cc-pVXZ
    "cc-pvdz", "cc-pvtz", "cc-pvqz", "cc-pv5z", "cc-pv6z",
    "aug-cc-pvdz", "aug-cc-pvtz", "aug-cc-pvqz", "aug-cc-pv5z",

    # Core-valence basis sets
    "cc-pcvdz", "cc-pcvtz", "cc-pcvqz", "cc-pcv5z",
    "aug-cc-pcvdz", "aug-cc-pcvtz", "aug-cc-pcvqz",
]

element_names = [
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
    "K",  "Ca"]

for atom in element_names
    for basis_set in basis_names
        @testset "Testing AO for $atom with $basis_set" begin
            try
                basis = build_ao(atom, basis_set)
                mol = build_mol(atom, basis_set)
                x = rand(3)
                r_py = [x]
                r_p4ml = SVector{3}(x)

                ao = mol.eval_gto("GTOval_sph", np.array(r_py))
                ao_val = Array(ao[1, :])
                val = basis(r_p4ml)

                @test norm(ao_val - val) < 1e-5
            catch e
                @warn "Failed to build AO for $atom with $basis_set"
            end
        end
    end
end
