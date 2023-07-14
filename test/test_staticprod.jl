
using Polynomials4ML, StaticArrays, ForwardDiff, Test, LinearAlgebra
using ACEbase.Testing: fdtest, println_slim, print_tf
P4ML = Polynomials4ML

##

@info("Testing _prod_grad - old implementation")

prodgrad = P4ML._prod_grad

for N = 1:5
   for ntest = 1:10
      local g, b, g1 
      b = rand(SVector{N,Float64})
      g = prodgrad(b.data, Val(N))
      g1 = ForwardDiff.gradient(prod, b)
      print_tf(@test g1 ≈ SVector(g...))
   end
end
println()

##

@info("Testing _static_prod - new implementation")

for ORD = 1:5 
   for ntest = 1:10 
      local b, ∂, p1, g1, u1, p2, p3, g3, p4, g4, u4 
      b = randn(SVector{ORD,Float64}) 
      ∂ = randn(SVector{ORD,Float64})
      p1 = prod(b)
      g1 = prodgrad(b.data, Val(ORD))
      u1 = ForwardDiff.gradient(_b -> dot(SVector(prodgrad(_b.data, Val(ORD))...), ∂), b).data
      p2 = P4ML._static_prod(b.data)
      p3, g3 = P4ML._grad_static_prod(b.data)
      p4, g4, u4 = P4ML._pb_grad_static_prod(∂.data, b.data)
      print_tf(@test p1 ≈ p2 ≈ p3 ≈ p4)
      print_tf(@test all(g1 .≈ g3 .≈ g4))
      print_tf(@test all(u1 .≈ u4))
   end
end
println() 
