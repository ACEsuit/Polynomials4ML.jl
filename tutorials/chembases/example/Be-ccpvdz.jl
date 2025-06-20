using chembases
using StaticArrays
using Test
using LinearAlgebra

atom = "Be"
basis_set = "cc-pvdz"
basis = build_ao(atom, basis_set)

# Compare with PySCF output at a random point
mol = build_mol(atom, basis_set)
x = rand(3)
r_py = [x]                      # For PySCF
r_p4ml = SVector{3}(x)          # For P4ML

using PyCall
np = pyimport("numpy")
ao = mol.eval_gto("GTOval_sph", np.array(r_py))
ao_val = Array(ao[1, :])
val = basis(r_p4ml)

@test norm(ao_val - val) < 1e-13

# Save the basis to a JSON file
save_all_bases_to_json(["Be"], ["cc-pvdz"], "ao_basis.json")

# Load AO basis from file
basis = load_basis_from_json("ao_basis.json", "Be", "cc-pvdz")

# Evaluate at a point
x = SVector{3}(rand(3))
val1 = basis(x)

# Compare with freshly built basis
basis2 = build_ao("Be", "cc-pvdz")
val2 = basis2(x)

@test norm(val1 - val2) < 1e-10