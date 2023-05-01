using LinearAlgebra, StaticArrays, Test, Printf
using Polynomials4ML, Polynomials4ML.Testing
using Polynomials4ML: evaluate, evaluate_d, evaluate_ed 
using Polynomials4ML.Testing: print_tf, println_slim 
using ForwardDiff

@info("Testing GaussianBasis")
n1 = 5 # degree
n2 = 3 
Pn = Polynomials4ML.legendre_basis(n1+1)
spec = [(n1 = n1, n2 = n2, l = l) for n1 = 1:n1 for n2 = 1:n2 for l = 0:n1-1] 
Dn = GaussianBasis()
ζ = rand(length(spec))
bRnl = AtomicOrbitalsRadials(Pn, Dn, spec, ζ) 
rr = 2 * rand(10) .- 1
Rnl = evaluate(bRnl, rr)
Rnl1, dRnl1 = evaluate_ed(bRnl, rr)
Rnl2, dRnl2, ddRnl2 = evaluate_ed2(bRnl, rr)


fdRnl = vcat([ ForwardDiff.derivative(r -> evaluate(bRnl, [r,]), r) 
               for r in rr ]...) 
fddRnl = vcat([ ForwardDiff.derivative(r -> evaluate_ed(bRnl, [r,])[2], r)
               for r in rr ]...) 

@show Rnl ≈ Rnl1 ≈ Rnl2 
@show dRnl1 ≈ dRnl2 ≈ fdRnl
@show ddRnl2 ≈ fddRnl

@info("Testing SlaterBasis")
n1 = 5 # degree
n2 = 3 
Pn = Polynomials4ML.legendre_basis(n1+1)
spec = [(n1 = n1, n2 = n2, l = l) for n1 = 1:n1 for n2 = 1:n2 for l = 0:n1-1] 
Dn = SlaterBasis()
ζ = rand(length(spec))
bRnl = AtomicOrbitalsRadials(Pn, Dn, spec, ζ) 
rr = 2 * rand(10) .- 1
Rnl = evaluate(bRnl, rr)
Rnl1, dRnl1 = evaluate_ed(bRnl, rr)
Rnl2, dRnl2, ddRnl2 = evaluate_ed2(bRnl, rr)


fdRnl = vcat([ ForwardDiff.derivative(r -> evaluate(bRnl, [r,]), r) 
               for r in rr ]...) 
fddRnl = vcat([ ForwardDiff.derivative(r -> evaluate_ed(bRnl, [r,])[2], r)
               for r in rr ]...) 

@show Rnl ≈ Rnl1 ≈ Rnl2 
@show dRnl1 ≈ dRnl2 ≈ fdRnl
@show ddRnl2 ≈ fddRnl

@info("Testing STOBasis")
n1 = 5 # degree
n2 = 1
Pn = Polynomials4ML.legendre_basis(n1+1)
spec = [(n1 = n1, n2 = n2, l = l) for n1 = 1:n1 for n2 = 1:1 for l = 0:n1-1] 
M = 3
ζ = [rand(length(spec), M),rand(length(spec), M)]
Dn = STO_NG()
bRnl = AtomicOrbitalsRadials(Pn, Dn, spec, ζ) 
rr = 2 * rand(10) .- 1
Rnl = evaluate(bRnl, rr)
Rnl1, dRnl1 = evaluate_ed(bRnl, rr)
Rnl2, dRnl2, ddRnl2 = evaluate_ed2(bRnl, rr)

fdRnl = vcat([ ForwardDiff.derivative(r -> evaluate(bRnl, [r,]), r) 
               for r in rr ]...) 
fddRnl = vcat([ ForwardDiff.derivative(r -> evaluate_ed(bRnl, [r,])[2], r)
               for r in rr ]...) 

@show Rnl ≈ Rnl1 ≈ Rnl2 
@show dRnl1 ≈ dRnl2 ≈ fdRnl
@show ddRnl2 ≈ fddRnl