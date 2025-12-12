using BenchmarkTools
using LinearAlgebra, StaticArrays, Test, Printf
using Polynomials4ML
using Polynomials4ML: evaluate, evaluate_d, evaluate_ed, evaluate_ed2, evaluate_ed_dp
using ACEbase.Testing: fdtest
using Zygote
using ObjectPools
using Random
using LuxCore

import Polynomials4ML as P4ML 

@info("Batched Implementation")
@info("evaluate, evaluated, evaluated2")
n1 = 5 # degree
n2 = 1
Pn = Polynomials4ML.legendre_basis(n1+1)
spec = [(n1 = n1, n2 = n2, l = l) for n1 = 1:n1 for n2 = 1:1 for l = 0:n1-1] 
M = 3
ζ = (rand(2 * length(spec), M), rand(2 * length(spec), M))
Dn = STO_NG(ζ)
bRnl = AtomicOrbitalsRadials(Pn, Dn, spec) 
rr = 2 * rand(10) .- 1

release!(rr)
@btime $evaluate($bRnl, $rr)
@btime $evaluate_ed($bRnl, $rr)
@btime $evaluate_ed2($bRnl, $rr)


@info("benchmark gradient")
l = Polynomials4ML.STOLayer(bRnl)
ps, st = LuxCore.setup(MersenneTwister(1234), l)

@btime $l($rr, $ps, $st)
Zygote.pullback(x -> l(x, ps, st)[1], rr)

@btime $evaluate($Dn, $rr)
@btime $evaluate_ed($Dn, $rr)
@btime $evaluate_ed2($Dn, $rr)
 
@info("Naive Implementation")
rr = rand()
@btime $evaluate($bRnl, $rr)
@btime $evaluate_ed($bRnl, $rr)
@btime $evaluate_ed2($bRnl, $rr)

















