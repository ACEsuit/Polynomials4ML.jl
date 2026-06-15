module SpheriCartExt

# SpheriCart-specific glue for the `AtomicOrbitals` basis: the precise complex
# value type and the `_rand_*` fixtures that default the angular part to a
# SpheriCart `SolidHarmonics`. Everything else about `AtomicOrbitals` is generic
# over `basis.Ylm` (used via the ACEbase `evaluate` interface) and lives in
# Polynomials4ML core.

import Polynomials4ML as P4ML
import Polynomials4ML: _ylm_valtype, AtomicOrbitals, construct_basis, _specidx,
                       MonoBasis, GaussianDecay, SlaterDecay, AbstractDecayFunction,
                       _rand_basis, _rand_gaussian_basis, _rand_slater_basis,
                       _rand_sto_basis
using SpheriCart: SolidHarmonics, SphericalHarmonics,
                  ComplexSolidHarmonics, ComplexSphericalHarmonics
using StaticArrays

# precise value type for the complex harmonics; the real harmonics use the
# generic `_ylm_valtype` fallback in P4ML core (which returns `S`).
_ylm_valtype(::Union{ComplexSolidHarmonics, ComplexSphericalHarmonics},
             ::Type{<: SVector{3, S}}) where {S} = Complex{S}

# ---- test/dev fixtures: AtomicOrbitals with a default SpheriCart Ylm

function _rand_basis(N1=4, N2=3;
    K::Int=1,
    T::Type=Float64,
    decay_type::AbstractDecayFunction=GaussianDecay(),
    ζinit = () -> rand(T, N1 * N2 * N1^2, K),
    Dinit = () -> ones(T, N1 * N2 * N1^2, K))

    Pn = MonoBasis(N1 + 1)
    Ylm = SolidHarmonics(N1 - 1)
    spec_list = [(n1=n1, n2=n2, l=l, m=m) for n1 in 1:N1, n2 in 1:N2, l in 0:N1-1 for m in -l:l]
    spec = SVector{length(spec_list)}(spec_list)
    spec_ln = unique((n1=s.n1, n2=s.n2, l=s.l) for s in spec)
    Dn = construct_basis(ζinit(), Dinit(), decay_type, spec_ln)
    specidx = _specidx(spec, Pn, Dn, Ylm)

    return AtomicOrbitals{length(spec), typeof(Pn), typeof(Dn), typeof(Ylm)}(Pn, Dn, Ylm, spec, specidx)
end

_rand_gaussian_basis(N1=4, N2=3, T=Float64) = _rand_basis(N1, N2; T=T)

_rand_slater_basis(N1=4, N2=3, T=Float64) = _rand_basis(N1, N2; T=T, decay_type = SlaterDecay())

_rand_sto_basis(N1=4, N2=2, K=4, T=Float64) = _rand_basis(N1, N2; T=T, K=K,
        ζinit = () -> rand(T, N1 * N2 * N1^2, K),
        Dinit = () -> rand(T, N1 * N2 * N1^2, K))

end
