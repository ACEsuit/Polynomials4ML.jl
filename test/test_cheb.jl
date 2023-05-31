using Polynomials4ML, Test
using Polynomials4ML: evaluate, evaluate_d, evaluate_dd
using Polynomials4ML.Testing: println_slim, print_tf, test_derivatives


##

@info("Testing Real Chebyshev Polynomials (ChebBasis)")
N = 10
basis = ChebBasis(N) 

@info("      correctness")
mm = natural_indices(basis)
print_tf(@test mm == 0:N-1)

for ntest = 1:30
   θ = 2*π * rand()
   x = cos(θ)
   P = basis(x)
   P2 = [ cos(m*θ) for m in mm ]
   print_tf(@test P ≈ P2)
end
println() 


##

@info("      test derivatives")
generate_x = () -> 2*rand()-1
test_derivatives(basis, generate_x)
