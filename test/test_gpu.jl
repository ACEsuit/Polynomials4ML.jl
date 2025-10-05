# This test script is not part of the test suite but can be run locally 
# to test that all basis sets evaluate correctly with GPU arrays as 
# inputs.  

using Polynomials4ML
import Polynomials4ML as P4ML 

# For Metal: 
using Metal
GPUArray = MtlArray
TFL = Float32

# or CUDA or ...

##

# stolen from test_lux.jl 
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

   X = [ Float32.(P4ML._generate_input(basis)) for _ = 1:nX ]
   X_dev = GPUArray(X)

   # standard CPU evaluation 
   P1, dP1 = P4ML.evaluate_ed(basis, X)

   # KA evaluation on CPU 
   P2 = similar(P1)
   P4ML.ka_evaluate!(P2, basis, X)
   P2a = similar(P1)
   dP2a = similar(dP1)
   P4ML.ka_evaluate_ed!(P2a, dP2a, basis, X)
   
   # KA evaluation on GPU
   P3_dev = GPUArray(P1)
   P4ML.ka_evaluate!(P3_dev, basis, X_dev)
   P3 = Array(P3_dev)
   P3a_dev = GPUArray(P1) 
   dP3a_dev = GPUArray(dP1) 
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
   
