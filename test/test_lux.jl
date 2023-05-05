using Polynomials4ML, Test, StaticArrays, Lux 
using Polynomials4ML: lux
using Random: default_rng
using ACEbase.Testing: println_slim
rng = default_rng()

##

@info("Testing Lux Layers for basic basis sets")

test_bases = [ (chebyshev_basis(10), () -> rand()), 
          (RTrigBasis(10), () -> rand()), 
          (CTrigBasis(10), () -> rand()), 
          (MonoBasis(10), ()-> rand()), 
          (legendre_basis(10), () -> rand()), 
          (CYlmBasis(5), () -> randn(SVector{3, Float64})), 
          (RYlmBasis(5), () -> randn(SVector{3, Float64})) ]

for (basis, rnd) in test_bases 
   x = rnd() 
   B1 = evaluate(basis, x)
   l = lux(basis)
   ps, st = Lux.setup(rng, l)
   B2 = l(x, ps, st)
   println_slim(@test B1 == parent(B2))
end
