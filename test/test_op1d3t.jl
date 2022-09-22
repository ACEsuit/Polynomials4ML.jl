
using Polynomials4ML, Test
using Polynomials4ML: evaluate, evaluate_d, evaluate_dd
using Polynomials4ML.Testing: println_slim, test_derivatives

@info("Testing OrthPolyBasis1D3T")


##

@info("Test consistency of evaluate_** functions")

N = 10
basis = OrthPolyBasis1D3T(randn(N), randn(N), randn(N))

test_derivatives(basis, () -> rand())

