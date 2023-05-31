using LinearAlgebra, StaticArrays, Test, Printf
using Polynomials4ML
using Polynomials4ML: evaluate, evaluate_d, evaluate_ed, legendre_basis, RYlmBasis, rand_sphere
using Polynomials4ML.Utils: gensparse
using Polynomials4ML.Testing: print_tf, println_slim 
using ForwardDiff
using ACEbase.Testing: fdtest
using Zygote
using Lux
using Random


P4ML = Polynomials4ML
rng = Random.default_rng()

# simple Dot product layer with weight for testing
module M1
   using LuxCore, LinearAlgebra, Random 
   import LuxCore:  AbstractExplicitLayer, initialparameters, initialstates
   struct DotL <: AbstractExplicitLayer
      nin::Int
   end
   function (l::DotL)(x::AbstractVector{<: Number}, ps, st)
      return dot(x, ps.W), st
   end
   initialparameters(rng::AbstractRNG, l::DotL) = ( W = randn(rng, l.nin), )
   initialstates(rng::AbstractRNG, l::DotL) = NamedTuple()
end

## 
totdeg = 8
maxL = 3

# Radial embedding and spherical harmonics
Rn = legendre_basis(totdeg)
Ylm = RYlmBasis(maxL)
ν = 2

# Pooling and SparseProduct + n-corr 
spec1p = [(i, y) for i = 1:totdeg for y = 1:maxL]
bA = P4ML.PooledSparseProduct(spec1p)

# define n-corr spec
tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
admissible = bb -> ((length(bb) == 0) || (sum(b[1] - 1 for b in bb ) < totdeg)) # cannot use <= since we cannot approxiate poly basis corresponding to (2, 15) with (15)
filter = bb -> (length(bb) == 0 || sum(idx2lm(b[2])[1] for b in bb) <= maxL)
specAA = gensparse(; NU = ν, tup2b = tup2b, admissible = admissible, filter = filter, minvv = fill(0, ν), maxvv = fill(length(spec1p), ν), ordered = true)
spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]

# correlation layer
bAA = P4ML.SparseSymmProd(spec)


# wrapping into lux layers
l_Rn = P4ML.lux(Rn)
l_Ylm = P4ML.lux(Ylm)
l_bA = P4ML.lux(bA)
l_bAA = P4ML.lux(bAA)

# formming model with Lux Chain
_norm(x) = norm.(x)

l_xnx = Lux.Parallel(nothing; normx = WrappedFunction(_norm), x = WrappedFunction(identity))
l_embed = Lux.Parallel(nothing; Rn = l_Rn, Ylm = l_Ylm)


simpleacemodel = Chain(xnx = l_xnx, embed = l_embed, A = l_bA , AA = l_bAA, out = M1.DotL(length(bAA)))
ps, st = Lux.setup(rng, simpleacemodel)

bX = [ rand_sphere() for _ = 1:32 ] 
simpleacemodel(bX, ps, st)

F(X) = simpleacemodel(X, ps, st)[1]
dF(X) = Zygote.gradient(x -> Lux.apply(simpleacemodel, x, ps, st)[1], X)[1]
#(l, st_), pb = pullback(x -> Lux.apply(simpleacemodel, x, ps, st), bX)
# gs = pb((l, nothing))[1]


fdtest(F, dF, bX, verbose = true)

