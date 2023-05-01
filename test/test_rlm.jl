using LinearAlgebra, StaticArrays, Test, Printf
using Polynomials4ML, Polynomials4ML.Testing
using Polynomials4ML: SphericalCoords, 
                      dspher_to_dcart, cart2spher, spher2cart, rand_sphere
using Polynomials4ML: evaluate, evaluate_d, evaluate_ed 
using Polynomials4ML.Testing: print_tf, println_slim 

verbose = false

nX = 10
L = 20
X = [ rand_sphere() for i = 1:nX ]
@info("Testing consistency of Complex spherical harmonics")
basis = CYlmBasis(L)
basis1 = CRlmBasis(L, false)

Rnl = evaluate(basis,X)
Rnl1, dRnl1 = evaluate_ed(basis,X)

Ynl = evaluate(basis1,X)
Ynl1, dYnl1 = evaluate_ed(basis1,X)

@show Rnl ≈ Rnl1 ≈ Ynl ≈ Ynl1 
@show dRnl1 ≈ dYnl1 

@info("Testing consistency of Real spherical harmonics")
basis = RYlmBasis(L)
basis1 = RRlmBasis(L, false)

X = rand_sphere()
X = [ rand_sphere() for i = 1:nX ]

Rnl = evaluate(basis,X)
Rnl1, dRnl1 = evaluate_ed(basis,X)
ddRnl2 = Polynomials4ML.laplacian(basis,X)

Ynl = evaluate(basis1,X)
Ynl1, dYnl1 = evaluate_ed(basis1,X)
ddYnl2 = Polynomials4ML.laplacian(basis1,X)

@show Rnl ≈ Rnl1 ≈ Ynl ≈ Ynl1 
@show dRnl1 ≈ dYnl1 
@show ddRnl2 ≈ ddYnl2


#@info("Testing consistency of Complex solid harmonics")
#basis = CRlmBasis(L)
#basis1 = CRlmBasis(L, true)

#Rnl = evaluate(basis,X)
#Rnl1, dRnl1 = evaluate_ed(basis,X)

#Ynl = evaluate(basis1,X)
#Ynl1, dYnl1 = evaluate_ed(basis1,X)

#@show Rnl ≈ Rnl1 ≈ Ynl ≈ Ynl1 
#@show dRnl1 ≈ dYnl1 

#@info("Testing consistency of Real solid harmonics")
#basis = RRlmBasis(L)
#basis1 = RRlmBasis(L, true)

#Rnl = evaluate(basis,X)
#Rnl1, dRnl1 = evaluate_ed(basis,X)
#ddRnl2 = Polynomials4ML.laplacian(basis,X)

#Ynl = evaluate(basis1,X)
#Ynl1, dYnl1 = evaluate_ed(basis1,X)
#ddYnl2 = Polynomials4ML.laplacian(basis1,X)

#@show Rnl ≈ Rnl1 ≈ Ynl ≈ Ynl1 
#@show dRnl1 ≈ dYnl1 
#@show ddRnl2 ≈ ddYnl2






@info("quick performance test")
@info("Complex spherical harmonics, single input")
L = 10
nX = 32
X = rand_sphere()
basis = CYlmBasis(L)
basis1 = CRlmBasis(L, false)

using BenchmarkTools
using Polynomials4ML: release!
@info("eval_basis")
@btime ( Y = evaluate($basis, $X); release!(Y); ) 
@info("eval_basis1")
@btime ( Y = evaluate($basis1, $X); release!(Y); )
@info("eval_d_basis")
@btime begin Y, dY = evaluate_ed($basis, $X); release!(Y); release!(dY); end 
@info("eval_d_basis1")
@btime begin Y, dY = evaluate_ed($basis1, $X); release!(Y); release!(dY); end 

X = [ rand_sphere() for i = 1:nX ]
@info("Complex spherical harmonics, $nX inputs")
@info("eval_basis")
@btime ( Y = evaluate($basis, $X); release!(Y); )
@info("eval_basis1") 
@btime ( Y = evaluate($basis1, $X); release!(Y); ) 
@info("eval_d_basis")
@btime begin Y, dY = evaluate_ed($basis, $X); release!(Y); release!(dY); end 
@info("eval_d_basis1")
@btime begin Y, dY = evaluate_ed($basis1, $X); release!(Y); release!(dY); end 

L = 10
nX = 32
X = rand_sphere()
basis = RYlmBasis(L)
basis1 = RRlmBasis(L, false)

@info("Real spherical harmonics, single input")
@info("eval_basis")
@btime ( Y = evaluate($basis, $X); release!(Y); )
@info("eval_basis1")
@btime ( Y = evaluate($basis1, $X); release!(Y); ) 
@info("eval_d_basis")
@btime begin Y, dY = evaluate_ed($basis, $X); release!(Y); release!(dY); end 
@info("eval_d_basis1")
@btime begin Y, dY = evaluate_ed($basis1, $X); release!(Y); release!(dY); end 

X = [ rand_sphere() for i = 1:nX]
@info("Real spherical harmonics, $nX inputs")
@info("eval_basis")
@btime ( Y = evaluate($basis, $X); release!(Y); ) 
@info("eval_basis1")
@btime ( Y = evaluate($basis1, $X); release!(Y); ) 
@info("eval_d_basis")
@btime begin Y, dY = evaluate_ed($basis, $X); release!(Y); release!(dY); end 
@info("eval_d_basis1")
@btime begin Y, dY = evaluate_ed($basis1, $X); release!(Y); release!(dY); end 






#@info("Complex solid harmonics, single input")
#L = 10
#nX = 32
#X = rand_sphere()
#basis = CRlmBasis(L)
#basis1 = CRlmBasis1(L, true)

#@info("eval_basis")
#@btime ( Y = evaluate($basis, $X); release!(Y); ) 
#@info("eval_basis1")
#@btime ( Y = evaluate($basis1, $X); release!(Y); ) 
#@info("eval_d_basis")
#@btime begin Y, dY = evaluate_ed($basis, $X); release!(Y); release!(dY); end 
#@info("eval_d_basis1")
#@btime begin Y, dY = evaluate_ed($basis1, $X); release!(Y); release!(dY); end 

#@info("Complex solid harmonics, $nX inputs")
#X = [ rand_sphere() for i = 1:nX ]

#@info("eval_basis")
#@btime ( Y = evaluate($basis, $X); release!(Y); ) 
#@info("eval_basis1")
#@btime ( Y = evaluate($basis1, $X); release!(Y); ) 
#@info("eval_d_basis")
#@btime begin Y, dY = evaluate_ed($basis, $X); release!(Y); release!(dY); end 
#@info("eval_d_basis1")
#@btime begin Y, dY = evaluate_ed($basis1, $X); release!(Y); release!(dY); end 



#L = 10
#nX = 32
#X = rand_sphere()
#basis = RRlmBasis(L)
#basis1 = RRlmBasis1(L, true)
#@info("Real solid harmonics, single input")
#@info("eval_basis")
#@btime ( Y = evaluate($basis, $X); release!(Y); ) 
#@info("eval_basis1")
#@btime ( Y = evaluate($basis1, $X); release!(Y); ) 
#@info("eval_d_basis")
#@btime begin Y, dY = evaluate_ed($basis, $X); release!(Y); release!(dY); end 
#@info("eval_d_basis1")
#@btime begin Y, dY = evaluate_ed($basis1, $X); release!(Y); release!(dY); end 

#X = [ rand_sphere() for i = 1:nX]
#@info("Real solid harmonics, $nX inputs")
#@info("eval_basis")
#@btime ( Y = evaluate($basis, $X); release!(Y); ) 
#@info("eval_basis1")
#@btime ( Y = evaluate($basis1, $X); release!(Y); ) 
#@info("eval_d_basis")
#@btime begin Y, dY = evaluate_ed($basis, $X); release!(Y); release!(dY); end 
#@info("eval_d_basis1")
#@btime begin Y, dY = evaluate_ed($basis1, $X); release!(Y); release!(dY); end 
