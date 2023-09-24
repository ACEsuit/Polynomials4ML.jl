using BenchmarkTools
using LinearAlgebra, StaticArrays, Test, Printf
using Polynomials4ML
using Polynomials4ML: evaluate, evaluate_d, evaluate_ed, evaluate_ed2, evaluate_ed_dp
using ACEbase.Testing: fdtest
using Zygote
using ObjectPools
using Random
using LuxCore

P4ML = Polynomials4ML

@info("Batched Implementation")
@info("evaluate, evaluated, evaluated2, evaluated_dp")
n1 = 5 # degree
n2 = 3 
Pn = P4ML.legendre_basis(n1+1)
spec = [(n1 = n1, n2 = n2, l = l) for n1 = 1:n1 for n2 = 1:n2 for l = 0:n1-1] 
ζ = rand(length(spec))
Dn = SlaterBasis(ζ)
bRnl = AtomicOrbitalsRadials(Pn, Dn, spec) 
rr = 2 * rand(10) .- 1

release!(rr)
@btime $evaluate($bRnl, $rr)
@btime $evaluate_ed($bRnl, $rr)
@btime $evaluate_ed2($bRnl, $rr)
@btime $evaluate_ed_dp($bRnl, $rr)


@info("benchmark gradient")
l = Polynomials4ML.AORLayer(bRnl)
ps, st = LuxCore.setup(MersenneTwister(1234), l)

@btime $l($rr, $ps, $st)
Zygote.pullback(x -> l(x, ps, st)[1], rr)
Zygote.pullback(p -> l(rr, p, st)[1], ps)

@btime $evaluate($Dn, $rr)
@btime $evaluate_ed($Dn, $rr)
@btime $evaluate_ed2($Dn, $rr)
@btime $evaluate_ed_dp($Dn, $rr)
 
@info("Naive Implementation")
rr = rand()
@btime $evaluate($bRnl, $rr)
@btime $evaluate_ed($bRnl, $rr)
@btime $evaluate_ed2($bRnl, $rr)
@btime $evaluate_ed_dp($bRnl, $rr)







