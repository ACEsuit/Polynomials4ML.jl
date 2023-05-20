using Test
using Polynomials4ML.Testing: println_slim, print_tf
using Polynomials4ML: SparseProduct, evaluate, evaluate_ed, evaluate_ed2, test_evaluate, test_evaluate_ed, test_evaluate_ed2
using LinearAlgebra: norm
using Polynomials4ML
using ACEbase.Testing: fdtest

##
# NB = rand(collect(5:30))
NB = 3
N = [i * 4 for i = 1:NB]

B = [randn(N[i]) for i = 1:NB]
∂B = [randn(N[i]) for i = 1:NB]
∂∂B = [randn(N[i]) for i = 1:NB]

spec = sort([ Tuple([rand(1:N[i]) for i = 1:NB]) for _ = 1:100 ])

basis = SparseProduct(spec)


## 

@info("Test serial evaluation")

BB = Tuple(B)

A1 = test_evaluate(basis, BB)
A2 = evaluate(basis, BB)

println_slim(@test A1 ≈ A2 )

@info("Test serial evaluation_ed")
BB = Tuple(B)
∂BB = Tuple(∂B)

A = test_evaluate_ed(basis, BB, ∂BB)

AA = evaluate(basis, BB)
A1 = evaluate_ed(basis, BB, ∂BB)[1]
A2 = evaluate_ed(basis, BB, ∂BB)[2]

println_slim(@test A ≈ A2 )
println_slim(@test AA ≈ A1 )
##

@info("Test serial evaluation_d2")
BB = Tuple(B)
∂BB = Tuple(∂B)
∂∂BB = Tuple(∂∂B)

A = test_evaluate_ed2(basis, BB, ∂BB, ∂∂BB)

AA = evaluate(basis, BB)
dA = evaluate_ed(basis, BB, ∂BB)[2]
A1 = evaluate_ed2(basis, BB, ∂BB, ∂∂BB)[1]
A2 = evaluate_ed2(basis, BB, ∂BB, ∂∂BB)[2]
A3 = evaluate_ed2(basis, BB, ∂BB, ∂∂BB)[3]

println_slim(@test A ≈ A3 )
println_slim(@test AA ≈ A1 )
println_slim(@test dA ≈ A2 )
@info("Test batch evaluation")

nX = 64
bBB = Tuple([randn(nX, N[i]) for i = 1:NB])
bA1 = zeros(ComplexF64, nX, length(basis))

for j = 1:nX
    bA1[j, :] = evaluate(basis, Tuple([bBB[i][j, :] for i = 1:NB]))
end

bA2 = evaluate(basis, bBB)

println_slim(@test bA1 ≈ bA2)

@info("Test batch evaluate_ed")

nX = 64 
bBB = Tuple([randn(nX, N[i]) for i = 1:NB])
bdBB = Tuple([randn(nX, N[i]) for i = 1:NB])
A1 = zeros(ComplexF64, nX, length(basis))
bA1 = zeros(ComplexF64, nX, length(basis))

for j = 1:nX
    A1[j, :] = evaluate_ed(basis, Tuple([bBB[i][j, :] for i = 1:NB]), Tuple([bdBB[i][j, :] for i = 1:NB]))[1]
    bA1[j, :] = evaluate_ed(basis, Tuple([bBB[i][j, :] for i = 1:NB]), Tuple([bdBB[i][j, :] for i = 1:NB]))[2]
end

A2 = evaluate_ed(basis, bBB, bdBB)[1]
bA2 = evaluate_ed(basis, bBB, bdBB)[2]

println_slim(@test A1 ≈ A2)
println_slim(@test bA1 ≈ bA2)
## 
@info("Test batch evaluate_d2")

nX = 64 
bBB = Tuple([randn(nX, N[i]) for i = 1:NB])
bdBB = Tuple([randn(nX, N[i]) for i = 1:NB])
bddBB = Tuple([randn(nX, N[i]) for i = 1:NB])
A1 = zeros(ComplexF64, nX, length(basis))
bA1 = zeros(ComplexF64, nX, length(basis))
bbA1 = zeros(ComplexF64, nX, length(basis))

for j = 1:nX
    A1[j, :] = evaluate_ed2(basis, Tuple([bBB[i][j, :] for i = 1:NB]), Tuple([bBB[i][j, :] for i = 1:NB]), Tuple([bddBB[i][j, :] for i = 1:NB]))[1]
    bA1[j, :] = evaluate_ed2(basis, Tuple([bBB[i][j, :] for i = 1:NB]), Tuple([bdBB[i][j, :] for i = 1:NB]), Tuple([bddBB[i][j, :] for i = 1:NB]))[2]
    bbA1[j, :] = evaluate_ed2(basis, Tuple([bBB[i][j, :] for i = 1:NB]), Tuple([bdBB[i][j, :] for i = 1:NB]), Tuple([bddBB[i][j, :] for i = 1:NB]))[3]
end

A2 = evaluate_ed2(basis, bBB, bdBB, bddBB)[1]
bA2 = evaluate_ed2(basis, bBB, bdBB, bddBB)[2]
bbA2 = evaluate_ed2(basis, bBB, bdBB, bddBB)[3]

println_slim(@test A1 ≈ A2)
println_slim(@test bA1 ≈ bA2)
println_slim(@test bbA1 ≈ bbA2)

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