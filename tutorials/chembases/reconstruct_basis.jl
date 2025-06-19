include("construct_basis.jl")

# Load a specific basis from the JSON file and reconstruct the full AtomicOrbitals object
function load_basis_from_json(file::String, atom::String, bset::String)
    data = JSON.parsefile(file)       # Load full dictionary from JSON
    d = data[atom][bset]              # Access the specific atom and basis set entry

    # Convert ζ and D back to matrix format (SMatrix)
    ζ = permutedims(hcat(d["ζ"]...))'
    D = permutedims(hcat(d["D"]...))'

    # Convert list of dictionaries back into a static vector of NamedTuples
    spec = SVector{length(d["spec"])}(
        NamedTuple{(:n1, :n2, :l, :m)}((s["n1"], s["n2"], s["l"], s["m"])) for s in d["spec"]
    )

    # Reconstruct the remaining components of AtomicOrbitals
    lmax = maximum(getfield.(spec, :l))
    Pn = MonoBasis(maximum(getfield.(spec, :n1)) + 1)
    Ylm = real_solidharmonics(lmax)
    spec_ln = unique((n1 = s.n1, n2 = s.n2, l = s.l) for s in spec)
    Dn = P4ML.construct_basis(ζ, D, P4ML.GaussianDecay(), spec_ln)
    specidx = P4ML._specidx(spec, Pn, Dn, Ylm)

    # Return the reconstructed AtomicOrbitals object
    return P4ML.AtomicOrbitals{length(spec), typeof(Pn), typeof(Dn), typeof(Ylm)}(
        Pn, Dn, Ylm, spec, specidx
    )
end

# Load from JSON
basis = load_basis_from_json("all_ao_basis.json", "Be", "cc-pvdz")
x = SVector{3}(rand(3))  # Random evaluation point

# Build from scratch for comparison
atom = "Be"
basis_set = "cc-pvdz"
basis2 = build_ao(atom, basis_set)

# Verify numerical consistency between original and loaded basis
@test norm(basis(x) - basis2(x)) < 1e-10
