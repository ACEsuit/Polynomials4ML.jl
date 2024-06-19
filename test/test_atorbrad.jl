using LinearAlgebra, StaticArrays, Test, Printf
using Polynomials4ML, Polynomials4ML.Testing
using Polynomials4ML: evaluate, evaluate_d, evaluate_ed, evaluate_ed2
using Polynomials4ML.Testing: print_tf, println_slim 
using ForwardDiff
using ACEbase.Testing: fdtest
using Zygote

P4ML = Polynomials4ML

##

@info("Testing GaussianBasis")
n1 = 5 # degree
n2 = 3 
Pn = P4ML.legendre_basis(n1+1)
spec = [(n1 = n1, n2 = n2, l = l) for n1 = 1:n1 for n2 = 1:n2 for l = 0:n1-1] 
ζ = rand(length(spec))
Dn = GaussianBasis(ζ)
bRnl = AtomicOrbitalsRadials(Pn, Dn, spec) 

P4ML.Testing.test_evaluate_xx(bRnl)
P4ML.Testing.test_chainrules(bRnl)
@warn("There are some allocations in GaussianBasis - to be test!!!")
P4ML.Testing.test_withalloc(bRnl; allowed_allocs = 192)



##
# ----------------------------------------------------
#   the rest here tests some more specialized functionality. 

using LuxCore
using Random
using Zygote
rng = Random.default_rng()
G = Polynomials4ML.AORLayer(bRnl)
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
    Rnl = evaluate(bRnl, x)
    u = G(x,ps,st)
    F(t) = begin
        Dn = GaussianBasis(_rr(t))
        bRnl = AtomicOrbitalsRadials(Pn, Dn, spec)
        dot(u[1], evaluate(bRnl, x))
    end
    dF(t) = begin
        Dn = GaussianBasis(_rr(t))
        bRnl = AtomicOrbitalsRadials(Pn, Dn, spec)
        G = Polynomials4ML.AORLayer(bRnl)
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
#    bRnl = AtomicOrbitalsRadials(Pn, Dn, spec)
#    G = Polynomials4ML.AORLayer(bRnl)
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
    Rnl = evaluate(bRnl, rr)
    G = Polynomials4ML.AORLayer(bRnl)
    u = G(rr,ps,st)
    F(t) = dot(u[1], evaluate(bRnl, _rr(t)))
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

@info("Testing SlaterBasis")
n1 = 5 # degree
n2 = 3 
Pn = P4ML.legendre_basis(n1+1)
spec = [(n1 = n1, n2 = n2, l = l) for n1 = 1:n1 for n2 = 1:n2 for l = 0:n1-1] 
ζ = rand(length(spec))
Dn = SlaterBasis(ζ)
bRnl = AtomicOrbitalsRadials(Pn, Dn, spec) 

P4ML.Testing.test_evaluate_xx(bRnl)
P4ML.Testing.test_chainrules(bRnl)
@warn("There are some allocations in AOR - to be test!!!")
P4ML.Testing.test_withalloc(bRnl; allowed_allocs = 192)


##


using LuxCore
using Random
using Zygote
rng = Random.default_rng()
G = Polynomials4ML.AORLayer(bRnl)
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
    Rnl = evaluate(bRnl, x)
    u = G(x,ps,st)
    F(t) = begin
        Dn = SlaterBasis(_rr(t))
        bRnl = AtomicOrbitalsRadials(Pn, Dn, spec)
        dot(u[1], evaluate(bRnl, x))
    end
    dF(t) = begin
        Dn = SlaterBasis(_rr(t))
        bRnl = AtomicOrbitalsRadials(Pn, Dn, spec)
        G = Polynomials4ML.AORLayer(bRnl)
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
#    bRnl = AtomicOrbitalsRadials(Pn, Dn, spec)
#    G = Polynomials4ML.AORLayer(bRnl)
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
    Rnl = evaluate(bRnl, rr)
    G = Polynomials4ML.AORLayer(bRnl)
    u = G(rr,ps,st)
    F(t) = dot(u[1], evaluate(bRnl, _rr(t)))
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
bRnl = AtomicOrbitalsRadials(Pn, Dn, spec) 


P4ML.Testing.test_evaluate_xx(bRnl)
P4ML.Testing.test_chainrules(bRnl)
@warn("There are some allocations in AOR - to be test!!!")
P4ML.Testing.test_withalloc(bRnl; allowed_allocs = 192)


##
