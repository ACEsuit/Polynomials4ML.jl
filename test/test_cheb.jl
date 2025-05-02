import Polynomials4ML as P4ML
using Polynomials4ML
using Test
using Polynomials4ML: natural_indices
using Polynomials4ML.Testing: print_tf, test_all 

##

@info("Testing ChebBasis (Standard Chebyshev Polynomials)")
N = 10
basis = ChebBasis(N) 

@info("      correctness")
mm = natural_indices(basis)
print_tf(@test mm == [ (n=n,) for n = 0:N-1] )

for ntest = 1:30
   local θ, x
   θ = 2*π * rand()
   x = cos(θ)
   P = basis(x)
   P_ref = [ cos(m.n*θ) for m in mm ]
   print_tf(@test P ≈ P_ref)
end
println() 

##

test_all(basis)
