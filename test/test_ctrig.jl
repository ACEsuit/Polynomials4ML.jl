

using Polynomials4ML, Test
using Polynomials4ML: natural_indices
using Polynomials4ML.Testing: print_tf, test_all 


##

@info("Testing Complex Trigonometric Polynomials (CTrigBasis)")
N = 10
basis = CTrigBasis(N) 

@info("      correctness")
mm = natural_indices(basis)
for ntest = 1:10
   local x 
   x = 2*π * rand()
   P = basis(x)
   P_ref = [ exp(im * m.n * x) for m in mm ]
   print_tf(@test P ≈ P_ref)
end
println() 

##

test_all(basis)
