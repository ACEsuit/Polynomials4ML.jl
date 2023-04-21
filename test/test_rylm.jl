

using LinearAlgebra, StaticArrays, Test, Printf
using Polynomials4ML, Polynomials4ML.Testing
using Polynomials4ML: SphericalCoords, index_y, 
                      dspher_to_dcart, cart2spher, spher2cart, rand_sphere
using Polynomials4ML: evaluate, evaluate_d, evaluate_ed 
using Polynomials4ML.Testing: print_tf, println_slim 

verbose = false

##

@info("Testing consistency of Real and Complex SH; Condon-Shortley convention")

function test_r2c(L, cY, rY)
   cYt = similar(cY)
   for l = 0:L
      m = 0
      i = index_y(l, m)
      cYt[i] = rY[i]
      for m = 1:l
         i_p = index_y(l, m)
         i_m = index_y(l, -m)
         # test the expressions
         #  Y_l^m    =      1/√2 (Y_{lm} - i Y_{l,-m})
         #  Y_l^{-m} = (-1)^m/√2 (Y_{lm} + i Y_{l,-m})
         cYt[i_p] = (1/sqrt(2)) * (rY[i_p] - im * rY[i_m])
         cYt[i_m] = (-1)^m * (1/sqrt(2)) * (rY[i_p] + im * rY[i_m])
      end
   end
   return cY ≈ cYt
end

maxL = 20
cSH = CYlmBasis(maxL)
rSH = RYlmBasis(maxL)

for nsamples = 1:30
   local R 
   R = rand_sphere()
   cY = evaluate(cSH, R)
   rY = evaluate(rSH, R)
   print_tf(@test test_r2c(maxL, cY, rY))
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


##

@info("Test: check derivatives of real spherical harmonics")
for nsamples = 1:30
   local R, rSH, h 
   R = @SVector rand(3)
   rSH = RYlmBasis(5)
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

rSH = RYlmBasis(10)
X = [ rand_sphere() for i = 1:21 ]
Y0 = evaluate(rSH, X)
Y1, dY1 = evaluate_ed(rSH, X)
Y2 = similar(Y1); dY2 = similar(dY1)
for i = 1:length(X)
   Y2[i, :] = evaluate(rSH, X[i])
   dY2[i, :] = evaluate_ed(rSH, X[i])[2]
end
println_slim(@test Y0 ≈ Y1 ≈ Y2)
println_slim(@test dY1 ≈ dY2)


## -- check the laplacian implementation 

using LinearAlgebra: tr
using ForwardDiff
P4 = Polynomials4ML

function fwdΔ(rYlm, x)
   Y = evaluate(rYlm, x)
   nY = length(Y)
   _j(x) = ForwardDiff.jacobian(x -> evaluate(rSH, x), x)[:]
   _h(x) = reshape(ForwardDiff.jacobian(_j, x), (nY, 3, 3))
   H = _h(x)
   return [ tr(H[i, :, :]) for i = 1:nY ]
end

for x in X 
   ΔY = P4.laplacian(rSH, x)
   ΔYfwd = fwdΔ(rSH, x)
   print_tf(@test ΔYfwd ≈ ΔY)
end
println() 

@info("check batched laplacian")
ΔY1 = P4.laplacian(rSH, X)
ΔY2 = similar(ΔY1)
for (i, x) in enumerate(X)
   ΔY2[i, :] = P4.laplacian(rSH, x)
end
println_slim(@test ΔY1 ≈ ΔY2)


@info("check eval_grad_laplace")
Y1, dY1, ΔY1 = P4.eval_grad_laplace(rSH, X)
Y2, dY2 = evaluate_ed(rSH, X)
ΔY2 = P4.laplacian(rSH, X)
println_slim(@test Y1 ≈ Y2)
println_slim(@test dY1 ≈ dY2)
println_slim(@test ΔY1 ≈ ΔY2)


## quick performance test 

using BenchmarkTools
using Polynomials4ML: release!

maxL = 10 
nX = 32 
rSH = RYlmBasis(maxL)
cSH = CYlmBasis(maxL)
X = [ rand_sphere() for i = 1:nX ]

@info("quick performance test")
@info("Real, single input")
@btime ( Y = evaluate($rSH, $(X[1])); release!(Y); )
@info("Real, $nX inputs")
@btime ( Y = evaluate($rSH, $X); release!(Y); )
@info("Complex, $nX inputs")
@btime ( Y = evaluate($cSH, $X); release!(Y); )

@info("Real, grad, single input")
@btime begin Y, dY = evaluate_ed($rSH, $(X[1])); release!(Y); release!(dY); end 
@info("Real, grad, $nX inputs")
@btime begin Y, dY = evaluate_ed($rSH, $X); release!(Y); release!(dY); end
@info("Complex, $nX inputs")
@btime begin Y, dY = evaluate_ed($cSH, $X); release!(Y); release!(dY); end

@info("laplacian, batched")
@btime begin ΔY = $(P4.laplacian)($rSH, $X); release!(ΔY); end 

@info("eval_grad_laplace")
@btime begin Y, dY, ΔY = $(P4.eval_grad_laplace)($rSH, $X); release!(Y); release!(dY); release!(ΔY); end 

##
