import Polynomials4ML as P4ML
using Polynomials4ML
using Test
using Polynomials4ML: natural_indices
using Polynomials4ML.Testing: print_tf, test_all

##

@info("Testing BernsteinBasis (Standard Bernstein Polynomials)")
N = 20
basis = BernsteinBasis(N)

@info("correctness")
mm = natural_indices(basis)
print_tf(@test mm == [ (n=n,) for n = 0:N-1] )

for ntest = 1:30
   x = rand()  
   P = basis(x)
   P_ref = [binomial(N-1, k.n) * x^k.n * (1 - x)^(N-1 - k.n) for k in mm]
   print_tf(@test P ≈ P_ref)
end

test_all(basis)

