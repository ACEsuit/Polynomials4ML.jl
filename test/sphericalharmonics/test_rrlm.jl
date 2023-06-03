using LinearAlgebra, StaticArrays, Test, Printf
using Polynomials4ML, Polynomials4ML.Testing
using Polynomials4ML: SphericalCoords, index_y, 
                      dspher_to_dcart, cart2spher, spher2cart, rand_sphere
using Polynomials4ML: evaluate, evaluate_d, evaluate_ed 
using Polynomials4ML.Testing: print_tf, println_slim 
using ACEbase.Testing: fdtest

verbose = false

##
function explicit_rsh(X)
   x,y,z = X
   Y00 = 1
   Y1m1 = y
   Y10 = z
   Y11 = x
   Y2m2 = sqrt(3)*x*y
   Y2m1 = sqrt(3)*y*z
   Y20 = 1/2*(3*z^2-(x^2+y^2+z^2))
   Y21 = sqrt(3)*x*z
   Y22 = 1/2*sqrt(3)*(x^2-y^2)
   Y3m3 = 1/2*sqrt(5/2) * (3*x^2-y^2)*y
   Y3m2 = sqrt(15)*x*y*z
   Y3m1 = 1/2*sqrt(3/2)*(5*z^2-(x^2+y^2+z^2))*y
   Y30 = 1/2*(5*z^2 - 3*(x^2+y^2+z^2))*z
   Y31 = 1/2*sqrt(3/2)*(5*z^2-(x^2+y^2+z^2))*x
   Y32 = 1/2*sqrt(15)*(x^2-y^2)*z
   Y33 = 1/2*sqrt(5/2)*(x^2-3*y^2)*x
   Y4m4 = 1/2*sqrt(35)*(x^2-y^2)*x*y
   Y4m3 = 1/2*sqrt(35/2)*(3*x^2-y^2)*y*z
   Y4m2 = 1/2*sqrt(5)*(7*z^2-(x^2+y^2+z^2))*x*y
   Y4m1 = 1/2*sqrt(5/2)*(7*z^2 - 3*(x^2+y^2+z^2))*y*z
   Y40 = 1/8*(35*z^4-30*z^2*(x^2+y^2+z^2)+3*(x^2+y^2+z^2)^2)
   Y41 = 1/2*sqrt(5/2)*(7*z^2 - 3*(x^2+y^2+z^2))*x*z
   Y42 = 1/4*sqrt(5)*(7*z^2 - (x^2+y^2+z^2)) *(x^2-y^2)
   Y43 = 1/2*sqrt(35/2)*(x^2-3*y^2)*x*z
   Y44 = 1/8*sqrt(35)*(x^4-6*x^2*y^2+y^4)
   return [Y00, Y1m1, Y10, Y11, Y2m2, Y2m1, Y20, Y21, Y22,
            Y3m3, Y3m2, Y3m1, Y30, Y31, Y32, Y33, 
            Y4m4, Y4m3, Y4m2, Y4m1, Y40, Y41, Y42, Y43, Y44]
end

@info("Test: check real solid harmonics against explicit expressions")
nsamples = 30
for n = 1:nsamples
   local X
   θ = rand() * π
   φ = (rand()-0.5) * 2*π
   r = 0.1+rand()
   X = SVector(r*sin(θ)*cos(φ), r*sin(θ)*sin(φ), r*cos(θ))
   SH = RRlmBasis(4)
   Y = evaluate(SH, X)
   for l = 0:4
      for m = 1:l
         i_p = index_y(l, m)
         i_m = index_y(l, -m)
         Y[i_p] =  Y[i_p] * (-1)^m
         Y[i_m] = Y[i_m] * (-1)* (-1)^m
      end
   end
   Yex = explicit_rsh(X)
   print_tf((@test Y ≈ Yex))
end
println()

@info("Testing consistency of Real and Complex SH; Condon-Shortley convention")
function test_r2c(L, cY, rY)
   cYt = similar(cY)
   for l = 0:L
      m = 0
      i = index_y(l, m)
      cYt[i] = rY[i]/sqrt(4*pi/(2*l+1))
      for m = 1:l
         i_p = index_y(l, m)
         i_m = index_y(l, -m)
         # test the expressions
         #  Y_l^m    =      1/√2 (Y_{lm} - i Y_{l,-m})
         #  Y_l^{-m} = (-1)^m/√2 (Y_{lm} + i Y_{l,-m})
         cYt[i_p] = (1/sqrt(2)) * (rY[i_p] - im * rY[i_m])/sqrt(4*pi/(2*l+1))
         cYt[i_m] = (-1)^m * (1/sqrt(2)) * (rY[i_p] + im * rY[i_m])/sqrt(4*pi/(2*l+1))
      end
   end
   return cY ≈ cYt
end

L = 20
cSH = CRlmBasis(L)
rSH = RRlmBasis(L)

for nsamples = 1:30
   local R 
   R = rand_sphere()
   cY = evaluate(cSH, R)
   rY = evaluate(rSH, R)
   print_tf(@test test_r2c(L, cY, rY))
end
println()

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
   rSH = RRlmBasis(5)
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
rSH = RRlmBasis(10)
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
   _j(x) = ForwardDiff.jacobian(x -> evaluate(rYlm, x), x)[:]
   _h(x) = reshape(ForwardDiff.jacobian(_j, x), (nY, 3, 3))
   H = _h(x)
   return [ tr(H[i, :, :]) for i = 1:nY ]
end

for x in X 
   ΔY = P4.laplacian(rSH, x)
   ΔYfwd = fwdΔ(rSH, x)
   print_tf(@test norm(ΔYfwd ≈ ΔY) < 1e-12)
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

using Zygote
@info("Test rrule")
using LinearAlgebra: dot 
rSH = RRlmBasis(10)
for ntest = 1:30
   local X
   local Y
   local u
   
   X = [ rand_sphere() for i = 1:21 ]
   Y = [ rand_sphere() for i = 1:21 ]
   _x(t) = X + t * Y
   A = evaluate(rSH, X)
   u = randn(size(A))
   F(t) = dot(u, evaluate(rSH, _x(t)))
   dF(t) = begin
       val, pb = Zygote.pullback(rSH, _x(t))
       ∂BB = pb(u)[1] # pb(u)[1] returns NoTangent() for basis argument
       return sum( dot(∂BB[i], Y[i]) for i = 1:length(Y) )
   end
   print_tf(@test fdtest(F, dF, 0.0; verbose = false))
end
println()
# ## quick performance test 
# this needs to move to a benchmarksuite 

# using BenchmarkTools
# using Polynomials4ML: release!

# maxL = 10 
# nX = 32 
# rSH = RRlmBasis(maxL)
# cSH = CRlmBasis(maxL)
# X = [ rand_sphere() for i = 1:nX ]

# @info("quick performance test")
# @info("Real, single input")
# @btime ( Y = evaluate($rSH, $(X[1])); release!(Y); )
# @info("Real, $nX inputs")
# @btime ( Y = evaluate($rSH, $X); release!(Y); )
# @info("Complex, $nX inputs")
# @btime ( Y = evaluate($cSH, $X); release!(Y); )

# @info("Real, grad, single input")
# @btime begin Y, dY = evaluate_ed($rSH, $(X[1])); release!(Y); release!(dY); end 
# @info("Real, grad, $nX inputs")
# @btime begin Y, dY = evaluate_ed($rSH, $X); release!(Y); release!(dY); end
# @info("Complex, $nX inputs")
# @btime begin Y, dY = evaluate_ed($cSH, $X); release!(Y); release!(dY); end

# @info("laplacian, batched")
# @btime begin ΔY = $(P4.laplacian)($rSH, $X); release!(ΔY); end 

# @info("eval_grad_laplace")
# @btime begin Y, dY, ΔY = $(P4.eval_grad_laplace)($rSH, $X); release!(Y); release!(dY); release!(ΔY); end 
