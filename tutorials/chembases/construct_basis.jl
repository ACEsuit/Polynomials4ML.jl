using PyCall
using Test
using StaticArrays
using LinearAlgebra
using Polynomials4ML
using Polynomials4ML: evaluate, evaluate_ed
import Polynomials4ML as P4ML

gto = pyimport("pyscf.gto")
np = pyimport("numpy")

# Utility function: convert a vector of vectors into a matrix (row-padded with zeros)
function TSMAT(vv::Vector{<:Vector{T}}) where {T}
    nrow = length(vv)
    ncol = maximum(length.(vv))
    M = zeros(T, nrow, ncol)
    for i in 1:nrow
        vi = vv[i]
        for j in 1:length(vi)
            M[i, j] = vi[j]
        end
    end
    return M
end

# Normalization factor for radial part: rⁿ * exp(-α r²)
function gto_norm(n::Int, α::Float64)
    prefac = 2^(2n + 3) * factorial(n + 1) * (2α)^(n + 1.5)
    denom = factorial(2n + 2) * sqrt(π)
    sqrt(prefac / denom)
end

# Renumber n2 index: for each angular momentum l, renumber its n2 starting from 1
function renumber_n2_per_l(spec::Vector{<:NamedTuple})
    l_n2_maps = Dict{Int, Dict{Int, Int}}()
    new_spec = NamedTuple[]
    for s in spec
        l = s.l
        n2 = s.n2
        map = get!(l_n2_maps, l, Dict{Int, Int}())
        new_n2 = get!(map, n2) do
            length(map) + 1
        end
        push!(new_spec, (n1 = s.n1, n2 = new_n2, l = l, m = s.m))
    end
    return SVector{length(new_spec)}(new_spec)
end

# Parse basis shells from PySCF to ζ (exponents), D (coefficients), and spec (quantum numbers)
function basis_from_pyscf_shells(shells)
    ζ = Vector{Vector{Float64}}()
    D = Vector{Vector{Float64}}()
    spec = Vector{NamedTuple{(:n1, :n2, :l, :m), Tuple{Int, Int, Int, Int}}}()

    for shell in shells
        l = shell[1]
        entries = shell[2:end]

        exponents = Float64[]
        coeff_lists = []

        for row in entries
            α = row[1]
            cs = row[2:end]
            push!(exponents, α)
            push!(coeff_lists, [Float64(c) for c in cs])
        end

        n_contractions = length(coeff_lists[1])
        n_primitives = length(exponents)

        for j in 1:n_contractions
            coeffs_j = [coeff_lists[i][j] for i in 1:n_primitives]
            norm_coeffs_j = [gto_norm(l, exponents[i]) * coeffs_j[i] for i in 1:n_primitives]
            push!(ζ, exponents)
            push!(D, norm_coeffs_j)
            if l == 1
                for m in [1, -1, 0]
                    push!(spec, (n1 = 0, n2 = length(D), l = l, m = m))
                end
            else
                for m = -l:l
                    push!(spec, (n1 = 0, n2 = length(D), l = l, m = m))
                end
            end
        end
    end

    spec = renumber_n2_per_l(spec)
    return TSMAT(ζ), TSMAT(D), spec
end

# Determine if the atom has an open-shell configuration (used to set spin in PySCF)
function isopen_shell(atom::String)
    valence_e = Dict(
        "H" => 1, "He" => 2,
        "Li" => 3, "Be" => 4, "B" => 5, "C" => 6, "N" => 7, "O" => 8, "F" => 9, "Ne" => 10,
        "Na" => 11, "Mg" => 12, "Al" => 13, "Si" => 14, "P" => 15, "S" => 16, "Cl" => 17, "Ar" => 18,
        "K" => 19, "Ca" => 20
    )
    Z = valence_e[atom]
    return Z % 2 == 1
end

# Create a PySCF molecule
function build_mol(atom, basis_set)
    mol = gto.Mole()
    mol.atom = "$atom 0 0 0.0"
    mol.basis = basis_set
    mol.cart = false
    mol.spin = isopen_shell(atom) ? 1 : 0
    mol.build()
    return mol
end

# Build AtomicOrbitals from PySCF for a given atom and basis set
function build_ao_from_pyscf(atom::String, basis_set::String)
    mol = build_mol(atom, basis_set)
    basis_dict = mol["_basis"]
    shells = basis_dict.__getitem__(atom)
    if shells isa Matrix
        shells = [shells]
    end

    ζ, D, spec = basis_from_pyscf_shells(shells)

    lmax = maximum(getfield.(spec, :l))
    Ylm = real_solidharmonics(lmax)

    Pn = MonoBasis(maximum(getfield.(spec, :n1)) + 1)
    spec_ln = unique((n1 = s.n1, n2 = s.n2, l = s.l) for s in spec)
    Dn = P4ML.construct_basis(ζ, D, P4ML.GaussianDecay(), spec_ln)
    specidx = P4ML._specidx(spec, Pn, Dn, Ylm)

    return P4ML.AtomicOrbitals{length(spec), typeof(Pn), typeof(Dn), typeof(Ylm)}(
        Pn, Dn, Ylm, spec, specidx
    )
end

# Build AO and match PySCF normalization by comparing at a random point
function build_ao(atom::String, basis_set::String)
    basis = build_ao_from_pyscf(atom, basis_set)
    mol = build_mol(atom, basis_set)

    x = rand(3)
    r_py = [x]
    r_p4ml = SVector{3}(x)

    ao = mol.eval_gto("GTOval_sph", np.array(r_py))
    ao_val = Array(ao[1, :])
    val = basis(r_p4ml)

    # Compute correction factors for mismatch in AO amplitude
    D_norm = ao_val ./ val
    idx = findall(x -> abs(x-1) > 1e-5, D_norm)

    DnD = Matrix(basis.Dn.D)
    for (i, b) in enumerate(idx)
        bb = basis.specidx[b][2]
        DnD[bb, :] = basis.Dn.D[bb, :] * D_norm[idx[i]]    
    end

    Dn = P4ML.construct_basis(basis.Dn.ζ, DnD, P4ML.GaussianDecay(), basis.Dn.spec)
    basis2 = P4ML.AtomicOrbitals{length(basis.spec), typeof(basis.Pn), typeof(Dn), typeof(basis.Ylm)}(
        basis.Pn, Dn, basis.Ylm, basis.spec, basis.specidx
    )

    val2 = basis2(r_p4ml)
    @assert norm(ao_val - val2) < 1e-5 "AO values do not match PySCF output"
    return basis2
end

