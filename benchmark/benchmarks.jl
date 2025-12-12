using Polynomials4ML
using BenchmarkTools
using LuxCore, Random, Zygote

import Polynomials4ML as P4ML 


SUITE = BenchmarkGroup()


## Test polynomials

SUITE["Polynomials"] = BenchmarkGroup()

N = 100
Np = 10
r = 2*rand(N) .- 1
tmp = zeros(N,N)
tmp_d = similar(tmp)
tmp_d2 = similar(tmp)

# Chebyshev
ch_basis = ChebBasis(Np)

SUITE["Polynomials"]["Chebyshev"] = BenchmarkGroup()
SUITE["Polynomials"]["Chebyshev"]["evaluation"] = @benchmarkable evaluate!($tmp, $ch_basis, $r)
SUITE["Polynomials"]["Chebyshev"]["derivative"] = @benchmarkable evaluate_ed!($tmp, $tmp_d, $ch_basis, $r)
SUITE["Polynomials"]["Chebyshev"]["2nd derivative"] = @benchmarkable evaluate_ed2!($tmp, $tmp_d, $tmp_d2, $ch_basis, $r)

# OrthPolyBasis1D3T

op_basis = OrthPolyBasis1D3T(randn(Np), randn(Np), randn(Np))

SUITE["Polynomials"]["OrtoPoly1d3"] = BenchmarkGroup()
SUITE["Polynomials"]["OrtoPoly1d3"]["evaluation"] = @benchmarkable evaluate!($tmp, $op_basis, $r)
SUITE["Polynomials"]["OrtoPoly1d3"]["derivative"] = @benchmarkable evaluate_ed!($tmp, $tmp_d, $op_basis, $r)
SUITE["Polynomials"]["OrtoPoly1d3"]["2nd derivative"] = @benchmarkable evaluate_ed2!($tmp, $tmp_d, $tmp_d2, $op_basis, $r)


## ACE pooling
# this is a copy from profile/ace/profile_sparseprodpool.jl

# Helpers
function _generate_basis(; order=3, len = 50)
    NN = [ rand(10:30) for _ = 1:order ]
    spec = sort([ ntuple(t -> rand(1:NN[t]), order) for _ = 1:len])
    return PooledSparseProduct(spec)
 end
 
 function _rand_input1(basis::PooledSparseProduct{ORDER}) where {ORDER} 
    NN = [ maximum(b[i] for b in basis.spec) for i = 1:ORDER ]
    BB = ntuple(i -> randn(NN[i]), ORDER)
 end
 
 function _rand_input(basis::PooledSparseProduct{ORDER}; nX = 10) where {ORDER} 
    NN = [ maximum(b[i] for b in basis.spec) for i = 1:ORDER ]
    BB = ntuple(i -> randn(nX, NN[i]), ORDER)
 end

# 

SUITE["ACE"] = BenchmarkGroup()
SUITE["ACE"]["SparceProduct"] = BenchmarkGroup()

order = 4
basis1 = _generate_basis(; order=order)
BB = _rand_input1(basis1)

nX = 64
order = 3
basis2 = _generate_basis(; order=order)
bBB = _rand_input(basis2; nX = nX)

SUITE["ACE"]["SparceProduct"]["no pooling"] = @benchmarkable evaluate($basis1, $BB)
SUITE["ACE"]["SparceProduct"]["pooling"]    = @benchmarkable evaluate($basis2, $bBB)

l = Polynomials4ML.lux(basis2)
ps, st = LuxCore.setup(MersenneTwister(1234), l)

SUITE["ACE"]["SparceProduct"]["lux evaluation"]  = @benchmarkable l($bBB, $ps, $st)
SUITE["ACE"]["SparceProduct"]["Zygote gradient"] = @benchmarkable Zygote.gradient(x -> sum($l(x, $ps, $st)[1]), $bBB)