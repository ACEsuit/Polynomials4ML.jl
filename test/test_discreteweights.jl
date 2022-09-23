
using Polynomials4ML, LinearAlgebra, Test 
using Polynomials4ML.Testing: print_tf 


function rand_basis(nX, N)
   xx = 2 * (rand(nX) .- 0.5) 
   ww = 1 .+ rand(nX)
   W = DiscreteWeights(xx, ww, :normalize)
   return orthpolybasis(N, W)
end

##

@info("check that the orthogonality relation is satisfied")

for ntest = 1:30 
   local N, basis, G 
   N = rand(5:20)
   nX = rand(100:300) 
   basis = rand_basis(nX, N)
   W = basis.meta["weights"]
   G = zeros(N, N)
   for (i, (x, w)) in enumerate(zip(W.X, W.W))
      P = basis(x)
      G += w * P * P'
   end

   print_tf( @test G ≈ I )
end
println() 

## 

@info("check that they are really polynomials")
@info("  ... TODO ... ")




