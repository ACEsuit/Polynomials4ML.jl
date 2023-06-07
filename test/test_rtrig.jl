using Polynomials4ML, Test
using Polynomials4ML: evaluate, evaluate_d, evaluate_dd
using Polynomials4ML.Testing: println_slim, print_tf, test_derivatives


##

@info("Testing Real Trigonometric Polynomials (RTrigBasis)")
N = 10
basis = RTrigBasis(N) 

@info("      correctness")
mm = natural_indices(basis)
for ntest = 1:30
   local x 
   x = 2*π * rand()
   P = basis(x)
   P2 = [ (m >= 0 ? cos(m*x) : sin(abs(m)*x)) for m in mm ]
   print_tf(@test P ≈ P2)
end
println() 


##

@info("      test derivatives")
generate_x = () -> rand()*2*π - π
test_derivatives(basis, generate_x)
