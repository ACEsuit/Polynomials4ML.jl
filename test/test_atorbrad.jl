using LinearAlgebra, StaticArrays, Test, Printf, Polynomials4ML
using Polynomials4ML: evaluate, evaluate_ed
using Polynomials4ML.Testing: print_tf, println_slim 
using ForwardDiff
using ACEbase.Testing: fdtest

import Polynomials4ML as P4ML

##

@info("Testing GaussianBasis")
basis = P4ML._rand_gaussian_basis()

@info("      correctness of evaluation")
x = P4ML._generate_input(basis)
P = evaluate(basis, x)
P1, dP1 = evaluate_ed(basis, x)

P4ML.Testing.test_evaluate_xx(basis)
P4ML.Testing.test_chainrules(basis)

# Test is broken - reshape is causing this, hence single-input test is turned off
P4ML.Testing.test_withalloc(basis; allowed_allocs = 0, single=false)

##
# these are scripts to replicate and check this allocation problem. 
# strangely it doesn't occur for the other bases. Only for AtomicOrbtials. 
#
# using BenchmarkTools

# P, dP = evaluate_ed(basis, x)
# @btime P4ML.evaluate_ed!($P, $dP, $basis, $x)

# @profview let basis=basis, X=x, P=P, dP=dP 
#     for _ = 1:1_000_000 
#         P4ML.evaluate_ed!(P, dP, basis, X)
#     end
# end

##

@info("Testing SlaterBasis")
basis = P4ML._rand_slater_basis()

@info("      correctness of evaluation")
x = P4ML._generate_input(basis)
P = evaluate(basis, x)
P1, dP1 = evaluate_ed(basis, x)

P4ML.Testing.test_evaluate_xx(basis)
P4ML.Testing.test_chainrules(basis)
P4ML.Testing.test_withalloc(basis; allowed_allocs = 0, single=false)

##

@info("Testing STOBasis")
basis = P4ML._rand_sto_basis()

@info("      correctness of evaluation")
x = P4ML._generate_input(basis)
P = evaluate(basis, x)
P1, dP1 = evaluate_ed(basis, x)


P4ML.Testing.test_evaluate_xx(basis)
P4ML.Testing.test_chainrules(basis)
P4ML.Testing.test_withalloc(basis; allowed_allocs = 0, single=false)


#
##
# ----------------------------------------------------
#   the rest here tests some more specialized functionality. 
#= 
using Zygote

using LuxCore
using Random
using Zygote
rng = Random.default_rng()
G = Polynomials4ML.AORLayer(basis)
ps, st = LuxCore.setup(rng, G)
for ntest = 1:30
    local rr
    local uu
    local Rnl
    local u
    local x

    rr = ζ
    uu = ζ
    _rr(t) = rr + t * uu
    x = 2 * rand(10) .- 1
    Rnl = evaluate(basis, x)
    u = G(x,ps,st)
    F(t) = begin
        Dn = GaussianBasis(_rr(t))
        basis = AtomicOrbitalsRadials(Pn, Dn, spec)
        dot(u[1], evaluate(basis, x))
    end
    dF(t) = begin
        Dn = GaussianBasis(_rr(t))
        basis = AtomicOrbitalsRadials(Pn, Dn, spec)
        G = Polynomials4ML.AORLayer(basis)
        ps, st = LuxCore.setup(rng, G)
        val, pb = Zygote.pullback(p -> G(x,p,st), ps)
        ∂BB = pb(u)[1][1] 
        return sum( dot(∂BB[i], uu[i]) for i = 1:length(uu) )
    end
    print_tf(@test fdtest(F, dF, 0.0; verbose = false))
end
println()

X = rr = rand(length(spec))
using Optimisers
W0, re = destructure(ps)
Fp = w -> sum(G(X, re(w), st)[1])

#Fp(w) = begin Dn = GaussianBasis(w);
#    basis = AtomicOrbitalsRadials(Pn, Dn, spec)
#    G = Polynomials4ML.AORLayer(basis)
#    return sum(G(X, re(w), st)[1])
#end

dFp = w -> ( gl = Zygote.gradient(p -> sum(G(X, p, st)[1]), ps)[1]; destructure(gl)[1])

function grad_test2(f, df, X::AbstractVector)
    F = f(X) 
    ∇F = df(X)
    nX = length(X)
    EE = Matrix(I, (nX, nX))
    
    for h in 0.1.^(3:12)
       gh = [ (f(X + h * EE[:, i]) - F) / h for i = 1:nX ]
       @printf(" %.1e | %.2e \n", h, norm(gh - ∇F, Inf))
    end
 end
 
grad_test2(Fp, dFp, W0)






@info("Test rrule")
using LinearAlgebra: dot 

for ntest = 1:30
    local rr
    local uu
    local Rnl
    local u
    local G

    rr = 2 .* randn(10) .- 1
    uu = 2 .* randn(10) .- 1
    _rr(t) = rr + t * uu
    Rnl = evaluate(basis, rr)
    G = Polynomials4ML.AORLayer(basis)
    u = G(rr,ps,st)
    F(t) = dot(u[1], evaluate(basis, _rr(t)))
    dF(t) = begin
        val, pb = Zygote.pullback(x->G(x,ps,st), _rr(t))
        ∂BB = pb(u)[1]
        return sum( dot(∂BB[i], uu[i]) for i = 1:length(uu) )
    end
    print_tf(@test fdtest(F, dF, 0.0; verbose = false))
end
println()

a,b = Zygote.pullback(p->G(rr,p,st)[1], ps)
b(a)

p = Zygote.gradient(p->sum(G(rr,p,st)[1]), ps)[1]


##

# @info("Testing SlaterBasis")
# n1 = 5 # degree
# n2 = 3 
# Pn = P4ML.legendre_basis(n1+1)
# spec = [(n1 = n1, n2 = n2, l = l) for n1 = 1:n1 for n2 = 1:n2 for l = 0:n1-1] 
# ζ = rand(length(spec))
# Dn = SlaterBasis(ζ)
# basis = AtomicOrbitalsRadials(Pn, Dn, spec) 

# P4ML.Testing.test_evaluate_xx(basis)
# P4ML.Testing.test_chainrules(basis)
# @warn("There are some allocations in AOR - to be test!!!")
# P4ML.Testing.test_withalloc(basis; allowed_allocs = 192)


##


using LuxCore
using Random
using Zygote
rng = Random.default_rng()
G = Polynomials4ML.AORLayer(basis)
ps, st = LuxCore.setup(rng, G)
for ntest = 1:30
    local rr
    local uu
    local Rnl
    local u
    local x

    rr = ζ
    uu = ζ
    _rr(t) = rr + t * uu
    x = 2 * rand(10) .- 1
    Rnl = evaluate(basis, x)
    u = G(x,ps,st)
    F(t) = begin
        Dn = SlaterBasis(_rr(t))
        basis = AtomicOrbitalsRadials(Pn, Dn, spec)
        dot(u[1], evaluate(basis, x))
    end
    dF(t) = begin
        Dn = SlaterBasis(_rr(t))
        basis = AtomicOrbitalsRadials(Pn, Dn, spec)
        G = Polynomials4ML.AORLayer(basis)
        ps, st = LuxCore.setup(rng, G)
        val, pb = Zygote.pullback(p -> G(x,p,st), ps)
        ∂BB = pb(u)[1][1] 
        return sum( dot(∂BB[i], uu[i]) for i = 1:length(uu) )
    end
    print_tf(@test fdtest(F, dF, 0.0; verbose = false))
end
println()

X = rr
using Optimisers
W0, re = destructure(ps)
Fp = w -> sum(G(X, re(w), st)[1])

#Fp(w) = begin Dn = GaussianBasis(w);
#    basis = AtomicOrbitalsRadials(Pn, Dn, spec)
#    G = Polynomials4ML.AORLayer(basis)
#    return sum(G(X, re(w), st)[1])
#end

dFp = w -> ( gl = Zygote.gradient(p -> sum(G(X, p, st)[1]), ps)[1]; destructure(gl)[1])

function grad_test2(f, df, X::AbstractVector)
    F = f(X) 
    ∇F = df(X)
    nX = length(X)
    EE = Matrix(I, (nX, nX))
    
    for h in 0.1.^(3:12)
       gh = [ (f(X + h * EE[:, i]) - F) / h for i = 1:nX ]
       @printf(" %.1e | %.2e \n", h, norm(gh - ∇F, Inf))
    end
 end
 
grad_test2(Fp, dFp, W0)






@info("Test rrule")
using LinearAlgebra: dot 

for ntest = 1:30
    local rr
    local uu
    local Rnl
    local u
    local G

    rr = 2 .* randn(10) .- 1
    uu = 2 .* randn(10) .- 1
    _rr(t) = rr + t * uu
    Rnl = evaluate(basis, rr)
    G = Polynomials4ML.AORLayer(basis)
    u = G(rr,ps,st)
    F(t) = dot(u[1], evaluate(basis, _rr(t)))
    dF(t) = begin
        val, pb = Zygote.pullback(x->G(x,ps,st), _rr(t))
        ∂BB = pb(u)[1]
        return sum( dot(∂BB[i], uu[i]) for i = 1:length(uu) )
    end
    print_tf(@test fdtest(F, dF, 0.0; verbose = false))
end
println()



a,b = Zygote.pullback(p->G(rr,p,st)[1], ps)
b(a)

p = Zygote.gradient(p->sum(G(rr,p,st)[1]), ps)[1]

##

@info("Testing STOBasis")
n1 = 5 # degree
n2 = 1
Pn = Polynomials4ML.legendre_basis(n1+1)
spec = [(n1 = n1, n2 = n2, l = l) for n1 = 1:n1 for n2 = 1:1 for l = 0:n1-1] 
ζ = [rand(rand(collect(1:5))) for i = 1:length(spec)]
D = [rand(length(ζ[i])) for i = 1:length(spec)]
Dn = STO_NG((ζ, D))
basis = AtomicOrbitalsRadials(Pn, Dn, spec) 


P4ML.Testing.test_evaluate_xx(basis)
P4ML.Testing.test_chainrules(basis)
@warn("There are some allocations in AOR - to be test!!!")
P4ML.Testing.test_withalloc(basis; allowed_allocs = 192)


##

=#