

using Polynomials4ML, Test
using Polynomials4ML: evaluate, evaluate_d, evaluate_dd, _generate_input
using Polynomials4ML.Testing: println_slim, test_evaluate_xx, print_tf, 
                              test_withalloc, test_chainrules
using LinearAlgebra: I, norm, dot 
# using QuadGK
using ACEbase.Testing: fdtest
P4ML = Polynomials4ML

@info("Testing Sobolev Basis")


##

maxn = 10
maxq = 30 

basis = P4ML.sobolev_basis(maxn; maxq = maxq)

test_evaluate_xx(basis)
test_withalloc(basis)
# test_chainrules(basis)

##
# some visual tests to keep around for the moment. 
#= 
using Plots

xx = range(-1, 1, length=200)
pL2 = evaluate(basis.basis, xx)
pH2 = evaluate(basis, xx)
signs = [1, -1, 1, -1, -1]
plt = plot() 
for n = 1:5
   plot!(plt, xx, pL2[:, n], label = "L2-$n", lw = 2, c = n)
   plot!(plt, xx, signs[n] * pH2[:, n], label = "H2-$n", lw = 2, c = n, ls = :dash)
end
plt

##

signs = [1, -1, 1, -1, -1, -1, -1, -1, 1, 1]
plt = plot(; ylims = (-1.3, 1.3)) 
for n = 6:10
   plot!(plt, xx, pL2[:, n], label = "L2-$n", lw = 2, c = n-5)
   plot!(plt, xx, signs[n] * pH2[:, n], label = "H2-$n", lw = 2, c = n-5, ls = :dash)
end
plt


n4 = (1:length(basis)).^4
@info("λ vs n^4")
display([ basis.meta["lambda"] n4 ])

=#
