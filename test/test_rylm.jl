

using LinearAlgebra, StaticArrays, Test, Printf
using Polynomials4ML, Polynomials4ML.Testing
using Polynomials4ML: SphericalCoords, index_y, 
                      dspher_to_dcart, cart2spher, spher2cart, rand_sphere
using Polynomials4ML: evaluate, evaluate_d, evaluate_ed 
using Polynomials4ML.Testing: print_tf, println_slim 

verbose = true

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

# quick performance test 

using BenchmarkTools

using Polynomials4ML: release!
maxL = 20 
nX = 64 
rSH = RYlmBasis(maxL)
X = [ rand_sphere() for i = 1:nX ]

@btime ( Y = evaluate($rSH, $(X[1])); release!(Y); )
@btime ( Y = evaluate($rSH, $X); release!(Y); )

##

# @info("Test: check derivatives of real spherical harmonics")
# for nsamples = 1:30
#    R = @SVector rand(3)
#    SH = RSHBasis(5)
#    Y, dY = evaluate_ed(SH, R)
#    DY = Matrix(transpose(hcat(dY...)))
#    errs = []
#    verbose && @printf("     h    | error \n")
#    for p = 2:10
#       h = 0.1^p
#       DYh = similar(DY)
#       Rh = Vector(R)
#       for i = 1:3
#          Rh[i] += h
#          DYh[:, i] = (evaluate(SH, SVector(Rh...)) - Y) / h
#          Rh[i] -= h
#       end
#       push!(errs, norm(DY - DYh, Inf))
#       verbose && @printf(" %.2e | %.2e \n", h, errs[end])
#    end
#    success = (minimum(errs[2:end]) < 1e-3 * maximum(errs[1:3])) || (minimum(errs) < 1e-10)
#    print_tf(@test success)
# end
# println()

