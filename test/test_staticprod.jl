
using Polynomials4ML, StaticArrays, ForwardDiff
P4ML = Polynomials4ML

##

@info("Testing _prod_grad - old implementation")

prodgrad = P4ML._prod_grad

for N = 1:5
   for ntest = 1:10
      local v1, g, b
      b = rand(SVector{N,Float64})
      g = prodgrad(b.data, Val(N))
      g1 = ForwardDiff.gradient(prod, b)
      print_tf(@test g1 ≈ SVector(g...))
   end
end
println()

@info("     _pb_prod_grad")
for ORDER = 1:5
   local val, pb, u, b 
   @info("order = $ORDER")
   pb_prodgrad = P4ML._pb_prod_grad
   b = rand(SVector{ORDER,Float64})
   g = prodgrad(b.data, Val(ORDER))
   ∂ = @SVector randn(length(g))
   val, pb = pb_prodgrad(∂.data, b.data, Val(ORDER))
   @test all(val .≈ g)
   u = randn(SVector{ORDER,Float64}) 
   println_slim(@test  fdtest( b -> sum(u .* prodgrad(tuple(b...), Val(ORDER))), 
            b -> [ pb_prodgrad(u.data, tuple(b...), Val(ORDER))[2]... ],
            [b...], verbose = false ) |> all )
end

##

