

using Polynomials4ML, Test
using Polynomials4ML: evaluate, evaluate_d, evaluate_dd
using Polynomials4ML.Testing: println_slim, test_derivatives

@info("Testing Complex Trigonometric Polynomials (CTrigBasis)")

using ForwardDiff
##

N = 10
basis = CTrigBasis(N) 
generate_x = () -> rand()*2*π - π
test_derivatives(basis, generate_x)
