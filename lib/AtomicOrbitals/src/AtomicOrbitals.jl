module AtomicOrbitals

# Quantum-chemistry atomic-orbital basis (`AtomicOrbitals = Pn * Dn * Ylm`),
# moved out of Polynomials4ML core. The angular part `Ylm` is a SpheriCart
# harmonics basis, used purely through the ACEbase `evaluate` interface (so it
# carries no P4ML parameters/state); the radial parts `Pn`, `Dn` are P4ML
# bases.

using StaticArrays, LinearAlgebra, Random
using Bumper, WithAlloc

import Polynomials4ML as P4ML
import Polynomials4ML: AbstractP4MLBasis, BATCH, MonoBasis,
                       evaluate, evaluate!, evaluate_ed, evaluate_ed!,
                       _evaluate!, _valtype, _generate_input,
                       _init_luxparams, _init_luxstate,
                       natural_indices, pullback_ps
# `_static_params` is owned by this package (not part of the P4ML interface)

import SpheriCart
using SpheriCart: SolidHarmonics, SphericalHarmonics,
                  ComplexSolidHarmonics, ComplexSphericalHarmonics

export AtomicOrbitals, RadialDecay, GaussianDecay, SlaterDecay

include("aorbasis.jl")

end
