using Test
using Polynomials4ML.Testing: println_slim, print_tf
using Polynomials4ML: SparseProduct, evaluate, evaluate_ed, evaluate_ed2, test_evaluate, test_evaluate_ed, test_evaluate_ed2
using LinearAlgebra: norm
using Polynomials4ML
using ACEbase.Testing: fdtest

##

N1 = 10
N2 = 20
N3 = 30

B1 = randn(N1)
B2 = randn(N2)
B3 = randn(N3)

∂B1 = randn(N1)
∂B2 = randn(N2)
∂B3 = randn(N3)

∂∂B1 = randn(N1)
∂∂B2 = randn(N2)
∂∂B3 = randn(N3)


spec = sort([ (rand(1:N1), rand(1:N2), rand(1:N3)) for i = 1:100 ])

basis = SparseProduct(spec)


## 

@info("Test serial evaluation")

BB = (B1, B2, B3)

A1 = test_evaluate(basis, BB)
A2 = evaluate(basis, BB)

println_slim(@test A1 ≈ A2 )

@info("Test serial evaluation_ed")
BB = (B1, B2, B3)
∂BB = (∂B1, ∂B2, ∂B3)

A1 = test_evaluate_ed(basis, BB, ∂BB)
A2 = evaluate_ed(basis, BB, ∂BB)[2]

println_slim(@test A1 ≈ A2 )
##

@info("Test serial evaluation_d2")
BB = (B1, B2, B3)
∂BB = (∂B1, ∂B2, ∂B3)
∂∂BB = (∂∂B1, ∂∂B2, ∂∂B3)

A1 = test_evaluate_ed2(basis, BB, ∂BB, ∂∂BB)
A2 = evaluate_ed2(basis, BB, ∂BB, ∂∂BB)[3]

println_slim(@test A1 ≈ A2 )

@info("Test batch evaluation")

nX = 64 
bBB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
bA1 = zeros(ComplexF64, nX, length(basis))

for j = 1:nX
    bA1[j, :] = evaluate(basis, (bBB[1][j, :], bBB[2][j, :], bBB[3][j, :]))
end

bA2 = evaluate(basis, bBB)

println_slim(@test bA1 ≈ bA2)

@info("Test batch evaluate_ed")

nX = 64 
bBB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
bdBB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
bA1 = zeros(ComplexF64, nX, length(basis))

for j = 1:nX
    bA1[j, :] = evaluate_ed(basis, (bBB[1][j, :], bBB[2][j, :], bBB[3][j, :]), (bdBB[1][j, :], bdBB[2][j, :], bdBB[3][j, :]))[2]
end

bA2 = evaluate_ed(basis, bBB, bdBB)[2]

println_slim(@test bA1 ≈ bA2)
## 
@info("Test batch evaluate_d2")

nX = 64 
bBB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
bdBB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
bddBB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
bA1 = zeros(ComplexF64, nX, length(basis))

for j = 1:nX
    bA1[j, :] = evaluate_ed2(basis, (bBB[1][j, :], bBB[2][j, :], bBB[3][j, :]), (bdBB[1][j, :], bdBB[2][j, :], bdBB[3][j, :]), (bddBB[1][j, :], bddBB[2][j, :], bddBB[3][j, :]))[3]
end

bA2 = evaluate_ed2(basis, bBB, bdBB, bddBB)[3]

println_slim(@test bA1 ≈ bA2)

@info("Testing _rrule_evaluate")
using LinearAlgebra: dot 

for ntest = 1:30
    local bBB
    local bUU
    local bA2
    bBB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
    bUU = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
    _BB(t) = ( bBB[1] + t * bUU[1], bBB[2] + t * bUU[2], bBB[3] + t * bUU[3] )
    bA2 = Polynomials4ML.evaluate(basis, bBB)
    u = randn(size(bA2))
    F(t) = dot(u, Polynomials4ML.evaluate(basis, _BB(t)))
    dF(t) = begin
        val, pb = Polynomials4ML._rrule_evaluate(basis, _BB(t))
        ∂BB = pb(u)
        return sum( dot(∂BB[i], bUU[i]) for i = 1:length(bUU) )
    end
    print_tf(@test fdtest(F, dF, 0.0; verbose=false))
end
println()

##