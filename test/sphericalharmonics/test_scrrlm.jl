using LinearAlgebra, StaticArrays, Test, Printf, SparseArrays, BlockDiagonals
using Polynomials4ML, Polynomials4ML.Testing
using Polynomials4ML: index_y, rand_sphere
using Polynomials4ML: evaluate, evaluate_d, evaluate_ed 
using Polynomials4ML.Testing: print_tf, println_slim 
using ACEbase.Testing: fdtest
using HyperDualNumbers: Hyper



verbose = false

@info("Testing consistency of Real and Complex SH; SpheriCart convention")

##

_maxL = 20
cSH = RRlmBasis(_maxL)
rSH = SCRRlmBasis(_maxL)

_R = rand_sphere()
_scale = evaluate(cSH, _R) ./ evaluate(rSH, _R)
for nsamples = 1:30
   local R 
   R = rand_sphere()
   cY = evaluate(cSH, R)
   rY = evaluate(rSH, R)
   print_tf(@test cY ./ rY ≈ _scale)
end
println()


##

@info("Check consistency of serial and batched evaluation")

X = [ rand_sphere() for i = 1:23 ]
Y1 = evaluate(rSH, X)
Y2 = similar(Y1) 
for i = 1:length(X)
   Y2[i, :] = evaluate(rSH, X[i])
end
println_slim(@test Y1 ≈ Y2)


##

@info("Test: check derivatives of real spherical harmonics")
for nsamples = 1:30
   local R, rSH, h 
   R = @SVector rand(3)
   rSH = SCRRlmBasis(5)
   Y, dY = evaluate_ed(rSH, R)
   DY = Matrix(transpose(hcat(dY...)))
   errs = []
   verbose && @printf("     h    | error \n")
   for p = 2:10
      h = 0.1^p
      DYh = similar(DY)
      Rh = Vector(R)
      for i = 1:3
         Rh[i] += h
         DYh[:, i] = (evaluate(rSH, SVector(Rh...)) - Y) / h
         Rh[i] -= h
      end
      push!(errs, norm(DY - DYh, Inf))
      verbose && @printf(" %.2e | %.2e \n", h, errs[end])
   end
   success = (minimum(errs[2:end]) < 1e-3 * maximum(errs[1:3])) || (minimum(errs) < 1e-10)
   print_tf(@test success)
end
println()


##

@info("Check consistency of serial and batched gradients")

rSH = SCRRlmBasis(10)
X = [ rand_sphere() for i = 1:21 ]

x2dualwrtj(x, j) = SVector{3}([Hyper(x[i], i == j, i == j, 0) for i = 1:3])

hX = [x2dualwrtj(x, 1) for x in X]


Y0 = evaluate(rSH, X)
Y1, dY1 = evaluate_ed(rSH, X)
Y2 = similar(Y1); dY2 = similar(dY1)
for i = 1:length(X)
   Y2[i, :] = evaluate(rSH, X[i])
   dY2[i, :] = evaluate_ed(rSH, X[i])[2]
end
println_slim(@test Y0 ≈ Y1 ≈ Y2)
println_slim(@test dY1 ≈ dY2)
