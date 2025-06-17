# This example demonstrates how to use the Polynomials4ML (P4ML) package to construct
# atomic orbitals based on standard quantum chemistry basis sets.
# Specifically, we download the cc-pvtz basis for beryllium (Be) from the
# Basis Set Exchange, extract its primitives and contraction coefficients, 
# and build a set of Gaussian-type atomic orbitals (GTOs)

using Polynomials4ML
using Polynomials4ML: evaluate, evaluate_ed
import Polynomials4ML as P4ML

using SpecialFunctions
using Downloads, JSON3

# Download basis set for a given atom
function download_basis(symbol::String, basis::String)
    url = "https://www.basissetexchange.org/api/basis/$basis/format/json?elements=$symbol&normalize=true"
    json_str = Downloads.download(url)
    return JSON3.read(read(json_str, String))
end

# Parse the shells into exponents, coefficients, and spec
function basis_functions(basis_data, Z::Int)
    shells = basis_data["elements"][string(Z)]["electron_shells"]
    ζ = Vector{Vector{Float64}}()
    D = Vector{Vector{Float64}}()
    spec = Vector{NamedTuple{(:n1, :n2, :l, :m), Tuple{Int, Int, Int, Int}}}()

    for shell in shells
        exponents = parse.(Float64, shell["exponents"])
        l = shell["angular_momentum"][1]
        for (idx, coeff) in enumerate(shell["coefficients"])
            coefficients = parse.(Float64, coeff)
            push!(ζ, exponents)
            push!(D, coefficients)
            for m = -l:l
                push!(spec, (n1 = 0, n2 = idx, l = l, m = m))
            end
        end
    end
    return ζ, D, spec
end

# Convert nested vectors to a matrix format
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

# EXAMPLE 1: Build the cc-pvtz basis set for Be (Z = 4)
basis_data = download_basis("Be", "cc-pvtz")
ζ, D, spec = basis_functions(basis_data, 4) # Z = 4 for Be

# Construct basis and combine them into `AtomicOrbitals`
Ylm = real_solidharmonics(maximum(getfield.(spec, :l)))
Pn = MonoBasis(maximum(getfield.(spec, :n1))+1)
ζ = TSMAT(ζ)
D = TSMAT(D)
spec_ln = unique((n1=s.n1, n2=s.n2, l=s.l) for s in spec)
Dn = P4ML.construct_basis(ζ, D, P4ML.GaussianDecay(), spec_ln)
specidx = P4ML._specidx(spec, Pn, Dn, Ylm)
basis = P4ML.AtomicOrbitals{length(spec), typeof(Pn), typeof(Dn), typeof(Ylm)}(Pn, Dn, Ylm, spec, specidx)

# Evaluate the Atomic Orbitals at a given `x`
using StaticArrays
x = SVector{3}([0.1, 0.2, 0.3])
evaluate(basis, x)