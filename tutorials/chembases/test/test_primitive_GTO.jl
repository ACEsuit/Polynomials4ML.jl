# This script tests the behavior of PySCF's GTO evaluation and demonstrates
# how primitive Gaussian-type orbitals (GTOs) relate to real spherical harmonics.
#
# It compares PySCF-evaluated spherical AOs with values computed
# using the GTO normalization factor, real solid harmonics, and Gaussian functions.
using PyCall
using StaticArrays
using Polynomials4ML
using Polynomials4ML: MonoBasis, real_solidharmonics
using Test
using LinearAlgebra
import SpheriCart: idx2lm, lm2idx

# Import PySCF modules via PyCall
np = pyimport("numpy")
gto = pyimport("pyscf.gto")

# GTO normalization factor for rⁿ * exp(-α r²)
function gto_norm(n, α)
    prefac = 2^(2n + 3) * factorial(n + 1) * (2α)^(n + 1.5)
    denom = factorial(2n + 2) * sqrt(π)
    sqrt(prefac / denom)
end

# Radial decay term: exp(-α |r|²)
radial(x, α) = exp(-α * sum(x .^ 2))

# Evaluation point
x = rand(3)
r_py = [x]                      # For PySCF (expects list of points)
r_p4ml = SVector{3}(x)          # For RealSolidHarmonics

α = rand()                         # Gaussian exponent

# Loop over angular momentum quantum numbers l = 0 to 6 (S to I orbitals)
for l in 0:6
    println("Testing l = $l")

    # Create a minimal PySCF Mole object with a single hydrogen atom and l-type GTO
    mol = gto.Mole()
    mol.atom = "H 0 0 0"
    mol.basis = Dict(
        "H" => gto.basis.parse("""
        H    $(["S", "P", "D", "F", "G", "H", "I"][l+1])
          $α   1.0
        """)
    )
    mol.cart = false
    mol.spin = 1
    mol.build()

    # Evaluate spherical AO values at given point
    ao = mol.eval_gto("GTOval_sph", np.array(r_py))  # shape: (1, 2l+1)
    ao_val = Array(ao[1, :])                         # Convert to Julia vector

    # Extract real spherical harmonics components Y_{l,-l} to Y_{l,l}
    Ylm = real_solidharmonics(l)(r_p4ml)[lm2idx(l,-l):end]

    # Compute the primitive GTO values using normalization × Ylm × radial decay
    yval = gto_norm(l, α) .* Ylm .* radial(x, α)

    # For l = 1 (P orbitals), reorder to match PySCF's output convention [pz, px, py]
    if l == 1
        yval = [yval[3] yval[1] yval[2]]'  # Reorder from [px, py, pz] to [pz, px, py]
    end

    # Test that custom GTO evaluation matches PySCF's result within tolerance
    @test norm(yval - ao_val) < 1e-8
end
