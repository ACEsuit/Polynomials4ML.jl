
using Polynomials4ML, Test, ForwardDiff
using Polynomials4ML: evaluate, evaluate_d, evaluate_dd, evaluate_ed, evaluate_ed2 
using Polynomials4ML.Testing: println_slim

@info("Testing OrthPolyBasis1D3T")


##

@info("Test consistency of evaluate_** functions")

N = 10
basis = OrthPolyBasis1D3T(randn(N), randn(N), randn(N))
x = rand() 

P1 = evaluate(basis, x)
P2, dP2 = evaluate_ed(basis, x)
dP3 = evaluate_d(basis, x)
P4, dP4, ddP4 = evaluate_ed2(basis, x)
ddP5 = evaluate_dd(basis, x)

println_slim(@test P1 ≈ P2 ≈ P4 )
println_slim(@test dP2 ≈ dP3 ≈ dP4)
println_slim(@test ddP4 ≈ ddP5)

##

@info("Test correctness of derivatives")

adP1 = ForwardDiff.derivative(x -> evaluate(basis, x), x)
println_slim(@test adP1 ≈ dP2)

addP3 = ForwardDiff.derivative(x -> evaluate_d(basis, x), x)
println_slim(@test addP3 ≈ ddP5)

##

@info("Test consistency of batched evaluation")

X = rand(16) 
bP1 = Polynomials4ML._alloc(basis, X)
bdP1 = Polynomials4ML._alloc(basis, X)
bddP1 = Polynomials4ML._alloc(basis, X)
for (i, x) in enumerate(X)
   bP1[i, :] = evaluate(basis, x)
   bdP1[i, :] = evaluate_d(basis, x)
   bddP1[i, :] = evaluate_dd(basis, x)
end

bP2 = evaluate(basis, X)
bP3, bdP3 = evaluate_ed(basis, X)
bP4, bdP4, bddP4 = evaluate_ed2(basis, X)

println_slim(@test bP2 ≈ bP1 ≈ bP3 ≈ bP4)
println_slim(@test bdP3 ≈ bdP1 ≈ bdP4)
println_slim(@test bddP4 ≈ bddP1)
