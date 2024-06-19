
# this is an experimental testset for working with batched 
# pooled products. This isn't really supported yet, and not 
# property working, hence not part of runtests. 
#
# Note also this hasn't yet been updated to the updated interface. 
#

@info("PooledSparseProduct - Multiple evaluations")

using BenchmarkTools, Test, Polynomials4ML
using ACEbase.Testing: println_slim, print_tf
using Polynomials4ML:  PooledSparseProduct, evaluate, 
                       evaluate!, evalpool!

function evalpool_multi!(A, bA, BBB)
   for i = 1:length(BBB)
      ACEcore.evalpool!(@view(A[:, i]), bA, BBB[i])
   end
   return nothing 
end

function pb_evalpool_multi!(∂BBB, ∂A, bA, BBB)
   for i = 1:length(BBB)
      ACEcore._pullback_evalpool!(∂BBB[i], @view(∂A[i, :]), bA, BBB[i])
   end
   return nothing 
end

##


N1 = 10 
N2 = 20 
N3 = 50 
spec = sort([ (rand(1:N1), rand(1:N2), rand(1:N3)) for i = 1:100 ])
basis = PooledSparseProduct(spec)

##

@info("Test batched evaluation")
nnX = [30, 33, 25, 13] 
sum_nnX = sum(nnX)
bBB1 = [ ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )  for nX in nnX]
bBB2 = tuple( [ vcat([bBB1[i][j] for i = 1:length(nnX)]...) for j=1:3 ]... )
target = vcat([ fill(i, nnX[i]) for i = 1:length(nnX)]...)

A1 = zeros(length(spec), length(nnX))
A2 = zeros(length(nnX), length(spec))

P4ML.evalpool_multi!(A1, basis, bBB1)
evalpool!(A2, basis, bBB2, target)

@info("Test evalpool ≈ evalpool_multi")
println_slim(@test A1' ≈ A2)

## 

∂A = randn(size(A2))
∂BB1 = deepcopy(bBB1)
∂BB2 = deepcopy(bBB2)

pb_evalpool_multi!(∂BB1, ∂A, basis, bBB1)
ACEcore._pullback_evalpool!(∂BB2, ∂A, basis, bBB2, target)

_∂BB1_ = tuple( [ vcat([∂BB1[i][j] for i = 1:length(nnX)]...) for j=1:3 ]... )

@info("pullback")
println_slim(@test all(_∂BB1_ .≈ ∂BB2))
