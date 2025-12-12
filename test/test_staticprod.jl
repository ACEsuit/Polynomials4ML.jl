
using Polynomials4ML, StaticArrays, ForwardDiff, Test, LinearAlgebra
using ACEbase.Testing: fdtest, println_slim, print_tf
import Polynomials4ML as P4ML

##

@info("Testing _static_prod")

for ORD = 1:5 
   for ntest = 1:10 
      local b, ∂, p1, g1, u1, p2, p3, g3, p4, g4, u4 
      b = randn(SVector{ORD,Float64}) 
      ∂ = randn(SVector{ORD,Float64})
      p1 = prod(b)
      _prodgrad(b) = ForwardDiff.gradient(prod, b)
      g1 = _prodgrad(b)
      h1 = ForwardDiff.hessian(prod, b)
      u1 = ForwardDiff.gradient(_b -> dot(_prodgrad(_b), ∂), b).data
      p2 = P4ML._static_prod(b.data)
      p3, g3 = P4ML._static_prod_ed(b.data)
      p4, g4, u4 = P4ML._pb_grad_static_prod(∂.data, b.data)
      p5, g5, h5 = P4ML._static_prod_ed2(b.data)
      print_tf(@test p1 ≈ p2 ≈ p3 ≈ p4 ≈ p5)
      print_tf(@test all(g1 .≈ g3 .≈ g4 .≈ g5))
      print_tf(@test all(u1 .≈ u4))
      print_tf(@test h1 ≈ h5)
   end
end
println() 

