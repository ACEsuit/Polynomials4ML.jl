import Polynomials4ML as P4ML
using Test
using Polynomials4ML: evaluate, evaluate_ed, natural_indices
using Polynomials4ML.Testing: println_slim, print_tf, 
                              test_evaluate_xx, 
                              test_withalloc, 
                              test_chainrules


##

@info("Testing ChebBasis (Standard Chebyshev Polynomials)")
N = 10
basis = ChebBasis(N) 


@info("      correctness")
mm = natural_indices(basis)
print_tf(@test mm == 0:N-1)

for ntest = 1:30
   local θ, x
   θ = 2*π * rand()
   x = cos(θ)
   P = basis(x)
   P_ref = [ cos(m*θ) for m in mm ]
   print_tf(@test P ≈ P_ref)
end
println() 

##

test_evaluate_xx(basis)
test_withalloc(basis)
# test_chainrules(basis)

##

# draft tests for kernelabstractions 
for _ = 1:10 # run 30 random tests 
   nX = rand(30:100)
   X = [ P4ML._generate_input(basis) for i = 1:nX ]
   P1, dP1 = evaluate_ed(basis, X)
   P2 = similar(P1)
   P3 = similar(P1)
   dP3 = similar(dP1)
   P4ML.ka_evaluate!(P2, basis, X)
   P4ML.ka_evaluate_ed!(P3, dP3, basis, X)
   print_tf(@test P1 ≈ P2 ≈ P3)
   print_tf(@test dP1 ≈ dP3)
end 

##
# ----------------------------------------------------- 
# TODO: move to orthopolybasis tests
# basis2 = chebyshev_basis(N; normalize=false)
# r = basis(x) ./ basis2(x)
   # P3 = basis2(x)
# test_evaluate_xx(basis2)
# test_withalloc(basis2)
# # test_chainrules(basis2)
