using BenchmarkTools, Test, Polynomials4ML, ChainRulesCore
using Polynomials4ML: PooledSparseProduct, evaluate, evaluate!
using LuxCore, Random, Zygote

import Polynomials4ML as P4ML 

##

function _generate_basis(; order=3, len = 50)
   NN = [ rand(10:30) for _ = 1:order ]
   spec = sort([ ntuple(t -> rand(1:NN[t]), order) for _ = 1:len])
   return PooledSparseProduct(spec)
end

function _rand_input1(basis::PooledSparseProduct{ORDER}) where {ORDER} 
   NN = [ maximum(b[i] for b in basis.spec) for i = 1:ORDER ]
   BB = ntuple(i -> randn(NN[i]), ORDER)
end

function _rand_input(basis::PooledSparseProduct{ORDER}; nX = rand(5:15)) where {ORDER} 
   NN = [ maximum(b[i] for b in basis.spec) for i = 1:ORDER ]
   BB = ntuple(i -> randn(nX, NN[i]), ORDER)
end

##

@info("Test evaluation with a single input (no pooling)")
order = 4
basis = _generate_basis(; order=order)
BB = _rand_input1(basis)
A2 = evaluate(basis, BB)
@btime $evaluate($basis, $BB)

## 

@info("Test pooling of multiple inputs")
nX = 64
order = 3
basis = _generate_basis(; order=order)
bBB = _rand_input(basis)
bA2 = evaluate(basis, bBB)
@btime $evaluate($basis, $bBB)

@info("benchmark gradient")
l = Polynomials4ML.lux(basis)
ps, st = LuxCore.setup(MersenneTwister(1234), l)

@btime $l($bBB, $ps, $st)

@btime $Zygote.gradient(x -> sum($l(x, $ps, $st)[1]), $bBB)


