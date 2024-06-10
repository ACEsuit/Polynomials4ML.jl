using Polynomials4ML, Test
using Polynomials4ML: evaluate, evaluate_d, evaluate_dd, 
                      natural_indices, index
using Polynomials4ML.Testing: println_slim, print_tf, 
                              test_derivatives, 
                              test_withalloc 


##

@info("Testing Real Chebyshev Polynomials (ChebBasis)")
N = 10
basis = ChebBasis(N) 
basis2 = chebyshev_basis(N; normalize=false)

@info("      correctness")
mm = natural_indices(basis)
print_tf(@test mm == 0:N-1)

θ = 2*π * rand()
x = cos(θ)
r = basis(x) ./ basis2(x)
for ntest = 1:30
   local θ
   local x
   θ = 2*π * rand()
   x = cos(θ)
   P = basis(x)
   P2 = [ cos(m*θ) for m in mm ]
   P3 = basis2(x)
   print_tf(@test P ≈ P2 && (P ./ P3 ≈ r))
end
println() 

##

@info("      test derivatives")
test_derivatives(basis)
test_derivatives(basis2)

##

@info("     test allocations")
test_withalloc(basis)
test_withalloc(basis2)

