module SpheriCartExt

# SpheriCart-specific glue for the `AtomicOrbitals` basis: the precise complex
# value type and the default angular basis (`SolidHarmonics`) used by the
# `_rand_*` example constructors. Everything else about `AtomicOrbitals` is
# generic over `basis.Ylm` (used via the ACEbase `evaluate` interface) and lives
# in Polynomials4ML core.

import Polynomials4ML: _ylm_valtype, _default_ylm
using SpheriCart: SolidHarmonics, ComplexSolidHarmonics, ComplexSphericalHarmonics
using StaticArrays

# precise value type for the complex harmonics; the real harmonics use the
# generic `_ylm_valtype` fallback in P4ML core (which returns `S`).
_ylm_valtype(::Union{ComplexSolidHarmonics, ComplexSphericalHarmonics},
             ::Type{<: SVector{3, S}}) where {S} = Complex{S}

# default angular basis for the `_rand_*` example constructors in P4ML core
_default_ylm(L) = SolidHarmonics(L)

end
