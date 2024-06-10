using Test
using Polynomials4ML.Testing: println_slim, print_tf
using Printf
using Polynomials4ML: SparseProduct, evaluate
using LinearAlgebra: norm
using Polynomials4ML
using ACEbase.Testing: fdtest
using Zygote
using HyperDualNumbers: Hyper

test_evaluate(basis::SparseProduct, BB::Tuple{Vararg{AbstractVector}}) = 
       [ prod(BB[j][basis.spec[i][j]] for j = 1:length(BB)) 
            for i = 1:length(basis) ]

test_evaluate(basis::SparseProduct, BB::Tuple) = 
       [ prod(BB[j][basis.spec[i][j]] for j = 1:length(BB)) 
            for i = 1:length(basis) ]

# test_evaluate(basis::SparseProduct, BB::Tuple{Vararg{AbstractMatrix}}) = 
#         [ test_evaluate(basis, ntuple(i -> BB[i][j, :], length(BB)))
#          for j = 1:size(BB[1], 1) )            


##
NB = 3 # For _rrule_evaluate test we need NB = 3, fix later by generalizing the test case
N = [i * 4 for i = 1:NB]
B = [randn(N[i]) for i = 1:NB]
hB = [Hyper.(bb, 1.0, 1.0, 0) for bb in B]

spec = sort([ Tuple([rand(1:N[i]) for i = 1:NB]) for _ = 1:6])
basis = SparseProduct(spec)

##

@info("Test serial evaluation")

BB = Tuple(B)
hBB = Tuple(hB)

A1 = test_evaluate(basis, BB)
A2 = evaluate(basis, BB)
hA2 = evaluate(basis, hBB)
hA2_val = [x.value for x in hA2]

println_slim(@test A1 ≈ A2 )
println_slim(@test A2 ≈ hA2_val )

##

@info("Test batch evaluation")
nX = 5
bBB = Tuple([randn(nX, N[i]) for i = 1:NB])
bA1 = zeros(ComplexF64, nX, length(basis))
for j = 1:nX
    bA1[j, :] = evaluate(basis, Tuple([bBB[i][j, :] for i = 1:NB]))
end

bA2 = evaluate(basis, bBB)

println_slim(@test bA1 ≈ bA2)

##

@info("Testing pullback")
using LinearAlgebra: dot 

N1 = 10
N2 = 20
N3 = 30

for ntest = 1:30
    local bBB
    local bUU
    local bA2
    local u
    bBB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
    bUU = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
    _BB(t) = ( bBB[1] + t * bUU[1], bBB[2] + t * bUU[2], bBB[3] + t * bUU[3] )
    bA2 = Polynomials4ML.evaluate(basis, bBB)
    u = randn(size(bA2))
    F(t) = dot(u, Polynomials4ML.evaluate(basis, _BB(t)))
    dF(t) = begin
        val, pb = Zygote.pullback(evaluate, basis, _BB(t))
        ∂BB = pb(u)[2] # pb(u)[1] returns NoTangent() for basis argument
        return sum( dot(∂BB[i], bUU[i]) for i = 1:length(bUU) )
    end
    print_tf(@test fdtest(F, dF, 0.0; verbose=false))
end
println()

##

# try with rrule
u, pb = Zygote.pullback(evaluate, basis, bBB)
ll = pb(u)


# TODO: look into why this is failing
# using ChainRulesTestUtils
# test_rrule(evaluate, basis, bBB)
