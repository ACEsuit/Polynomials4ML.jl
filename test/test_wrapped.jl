

using Polynomials4ML, LinearAlgebra, StaticArrays, Test, Random, 
      LuxCore, Lux
using Polynomials4ML: evaluate, evaluate_ed
using Polynomials4ML.Testing: print_tf, println_slim 
using ForwardDiff
using ForwardDiff: Dual 

import Polynomials4ML as P4ML
import ForwardDiff as FD

rng = Random.default_rng()

@info("Testing ChainedBasis")

##

struct LinL <: AbstractLuxLayer
   in_dim::Int 
   out_dim::Int 
end

LuxCore.initialparameters(rng::AbstractRNG, l::LinL) = 
      ( W = randn(rng, l.out_dim, l.in_dim), )

LuxCore.initialstates(rng::AbstractRNG, l::LinL) = NamedTuple()

(l::LinL)(x::AbstractVector, ps, st) = ps.W * x, st 

(l::LinL)(X::AbstractMatrix, ps, st) = X * transpose(ps.W), st 



##

struct ChainedBasis{TCH} 
   chain::TCH
end

P4ML.evaluate(b::ChainedBasis, x, ps, st) = 
      Lux.apply(b.chain, x, ps, st)


function P4ML.evaluate_ed(b::ChainedBasis, x::Number, ps, st)
   B_dB, st = P4ML.evaluate(b, Dual(x, one(x)), ps, st)
   B = FD.value.(B_dB) 
   dB = [bdb.partials[1] for bdb in B_dB] 
   return (B, dB), st
end

function P4ML.evaluate(b::ChainedBasis, X::AbstractVector{<: Number}, ps, st) 
   N = length(X)
   M = length(b.chain.basis)
   B = zeros(eltype(_valtype(b, eltype(X))), N, M) 
   for i in 1:N
      B[i, :], st = P4ML.evaluate(b, X[i], ps, st)
   end
   return B, st
end

##

basis = ChebBasis(5)
trans = x -> 1 ./ (1 .+ x)

# old approach 
tbasis = P4ML.TransformedBasis(trans, basis)
P4ML._generate_input(::typeof(tbasis)) = rand() 
ps0, st0 = LuxCore.setup(rng, tbasis)
X = [ P4ML._generate_input(tbasis) for _ in 1:10 ]
x = 0.5
b0 = evaluate(tbasis, x, ps0, st0)

# new approach 
chb = ChainedBasis(Chain(; trans=WrappedFunction(trans), basis=basis))
ps, st = LuxCore.setup(rng, chb.chain)
b1, _ = evaluate(chb, x, ps, st)
(b1a, db1a), _ = evaluate_ed(chb, 0.5, ps, st) 

b1 ≈ b0
b1a ≈ b0

##
# new new approach 
wrb = P4ML.wrapped_basis(
               Chain(; trans=WrappedFunction(trans), basis=basis), 
               1.0) 
b3 = evaluate(wrb, [x,], ps, st)[:]

(b3a, db3a) = evaluate_ed(wrb, [x,], ps, st)

b3 ≈ b1
b3a[:] ≈ b1 
db3a[:] ≈ db1a 



len_basis = length(basis)
wrb2 = P4ML.wrapped_basis(
                Chain(; trans = WrappedFunction(trans), 
                       basis = basis, 
                       linear = LinL(len_basis, len_basis ÷ 2) ), 
                1.0 )
ps, st = LuxCore.setup(rng, wrb2)

b4 = evaluate(wrb2, [x,], ps, st)
b4[:] ≈ ps.linear.W * basis(trans(x))

(b4a, db4a) = evaluate_ed(wrb2, [x,], ps, st)
