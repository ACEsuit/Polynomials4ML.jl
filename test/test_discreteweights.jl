
using Polynomials4ML, LinearAlgebra, Test 
using Polynomials4ML.Testing: print_tf 
using Polynomials4ML: DiscreteWeights

function rand_basis(nX, N)
   xx = 2 * (rand(nX) .- 0.5) 
   ww = 1 .+ rand(nX)
   W = DiscreteWeights(xx, ww, :normalize)
   return orthpolybasis(N, W), W
end

##

@info("check that the orthogonality relation is satisfied")

for ntest = 1:30 
   local N, basis, G 
   N = rand(5:20)
   nX = rand(30:100) 
   basis, W = rand_basis(nX, N)
   G = zeros(N, N)
   for (i, (x, w)) in enumerate(zip(W.X, W.W))
      P = basis(x)
      G += w * P * P'
   end

   print_tf( @test G ≈ I )
end
println() 
