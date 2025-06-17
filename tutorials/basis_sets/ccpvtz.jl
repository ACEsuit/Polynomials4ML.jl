# EXAMPLE: 
# BE CCPVTZ BASIS_SET: 

using Polynomials4ML
using Polynomials4ML: evaluate, evaluate_ed
import Polynomials4ML as P4ML

using SpecialFunctions
using Downloads, JSON3

function download_basis(symbol::String, basis::String)
    url = "https://www.basissetexchange.org/api/basis/$basis/format/json?elements=$symbol&normalize=true"
    json_str = Downloads.download(url)
    return JSON3.read(read(json_str, String))
end

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

basis_data = download_basis("Be", "cc-pvtz")
ζ, D, spec = basis_functions(basis_data, 4) # Z = 4 for Be

Ylm = real_solidharmonics(maximum(getfield.(spec, :l)))
Pn = MonoBasis(maximum(getfield.(spec, :n1))+1)
ζ = P4ML.TSMAT(ζ)
D = P4ML.TSMAT(D)
spec_ln = unique((n1=s.n1, n2=s.n2, l=s.l) for s in spec)
#for (i, b) in enumerate(spec_ln)
#   for j = 1:size(ζ, 2)
        #N = primitive_norm(ζ[i, j], b.l)
#        D[i, j] *= N
#   end
#end
Dn = P4ML.construct_basis(ζ, D, P4ML.GaussianDecay(), spec_ln)
specidx = P4ML._specidx(spec, Pn, Dn, Ylm)
basis = P4ML.AtomicOrbitals{length(spec), typeof(Pn), typeof(Dn), typeof(Ylm)}(Pn, Dn, Ylm, spec, specidx)

using StaticArrays
x = SVector{3}([0.1, 0.2, 0.3])
evaluate(basis, x)