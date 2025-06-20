using Polynomials4ML, Test
using Polynomials4ML: natural_indices
using Polynomials4ML.Testing: print_tf, test_all 

##

@info("Testing Real Trigonometric Polynomials (RTrigBasis)")
N = 10
basis = RTrigBasis(N) 

@info("      correctness")
mm = natural_indices(basis)
for ntest = 1:30
   local x, P
   x = 2*π * rand()
   P = basis(x)
   P2 = [ (m.n >= 0 ? cos(m.n*x) : sin(abs(m.n)*x)) for m in mm ]
   print_tf(@test P ≈ P2)
end
println() 

##

test_all(basis)
