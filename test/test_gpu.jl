# This test script is not part of the test suite but can be run locally 
# to test that all basis sets evaluate correctly with GPU arrays as 
# inputs.  

using Polynomials4ML, Random, LuxCore 
import Polynomials4ML as P4ML 

# For Metal: 
using Metal
dev = Metal.mtl 
TFL = Float32

# or CUDA or ...

##

test_bases = [ ChebBasis(8), 
               BernsteinBasis(8), 
               RTrigBasis(10), 
               CTrigBasis(10), 
               MonoBasis(10), 
               TFL(chebyshev_basis(10)),
               TFL(legendre_basis(10)),
               real_sphericalharmonics(5; T = TFL, static=true), 
               real_solidharmonics(5; T = TFL, static=true), 
               #complex_sphericalharmonics(5), 
               #complex_solidharmonics(5), 
               ]

##      

for basis in test_bases
   @info("Test KA evaluation for $(typeof(basis).name.name)")
   local P1, P2, x
   nX = rand(500:1500)

   X = [ TFL.(P4ML._generate_input(basis)) for _ = 1:nX ]
   X_dev = dev(X)

   # standard CPU evaluation 
   P1, dP1 = P4ML.evaluate_ed(basis, X)

   # KA evaluation on CPU 
   P2 = similar(P1)
   P4ML.ka_evaluate!(P2, basis, X)
   P2a = similar(P1)
   dP2a = similar(dP1)
   P4ML.ka_evaluate_ed!(P2a, dP2a, basis, X)
   
   # KA evaluation on GPU
   P3_dev = dev(P1)
   P4ML.ka_evaluate!(P3_dev, basis, X_dev)
   P3 = Array(P3_dev)
   P3a_dev = dev(P1) 
   dP3a_dev = dev(dP1) 
   P4ML.ka_evaluate_ed!(P3a_dev, dP3a_dev, basis, X_dev)
   P3a = Array(P3a_dev)
   dP3a = Array(dP3a_dev)

   # allocating GPU evaluation 
   P4_dev = basis(X_dev)
   P4a_dev = P4ML.evaluate(basis, X_dev) 
   P4b_dev, dP4b_dev = P4ML.evaluate_ed(basis, X_dev)
   P4 = Array(P4_dev)
   P4a = Array(P4a_dev)
   P4b = Array(P4b_dev)
   dP4b = Array(dP4b_dev)

   @show P1 ≈ P2 ≈ P3 ≈ P4 ≈ P2a ≈ P3a ≈ P4a ≈ P4b
   @show dP1 ≈ dP2a ≈ dP3a ≈ dP4b
end
   
## 

using Metal, Functors

_float32(nt::NamedTuple) = Functors.fmap(_float32, nt) 
_float32(x::AbstractArray) = _float32.(x)
_float32(x::AbstractFloat) = Float32(x)
_float32(x) = x

@info("Testing F32 GPU evaluation consistency via state transfer")

for basis in [ chebyshev_basis(10),
               legendre_basis(10),
               real_sphericalharmonics(5; static=true), 
               real_solidharmonics(5; static=true), 
               ]
   nX = rand(500:1500)
   X = [ P4ML._generate_input(basis) for _ = 1:nX ]
   X_32 = _float32(X)
   X_dev = dev(X_32)

   ps, st = LuxCore.setup(MersenneTwister(1234), basis)
   st_32 = _float32(st)
   ps_dev = dev(ps) 
   st_dev = dev(st_32)

   P1 = basis(X, ps, st)[1]
   P2 = basis(X_32, ps, st_32)[1] 
   P3_dev = basis(X_dev, ps_dev, st_dev)[1] 
   P3 = Array(P3_dev)
   @show P1 ≈ P2 ≈ P3

end
