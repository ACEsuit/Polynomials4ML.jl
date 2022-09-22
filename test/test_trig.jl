

using Polynomials4ML, Test
using Polynomials4ML: evaluate, evaluate_d, evaluate_dd
using Polynomials4ML.Testing: println_slim, print_tf, test_derivatives


##

@info("Testing Complex Trigonometric Polynomials (CTrigBasis)")
N = 10
basis = CTrigBasis(N) 

@info("      correctness")
mm = natural_indices(basis)
for ntest = 1:10
   x = 2*π * rand()
   P = basis(x)
   print_tf(@test all( 
            P[index(basis, m) ] ≈ exp(im * m * x)
            for m in mm ))
end

##

@info("      test derivatives")
generate_x = () -> rand()*2*π - π
test_derivatives(basis, generate_x)
