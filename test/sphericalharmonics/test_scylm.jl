using LinearAlgebra, StaticArrays, Test, Printf, SparseArrays, BlockDiagonals
using Polynomials4ML, Polynomials4ML.Testing
using Polynomials4ML: index_y, rand_sphere
using Polynomials4ML: evaluate, evaluate_d, evaluate_ed 
using Polynomials4ML.Testing: print_tf, println_slim 
# using ACEbase.Testing: fdtest
using HyperDualNumbers: Hyper



verbose = false

@info("Testing consistency of Real and Complex SH; SpheriCart convention")

# SpheriCart R2C transformation
function ctran3(L)
   AA = zeros(ComplexF64, 2L+1, 2L+1)
   for i = 1:2L+1
       for j in [i, 2L+2-i]
           AA[i,j] = begin 
               if i == j == L+1
                   1
               elseif i > L+1 && j > L+1
                   (-1)^(i-L-1)/sqrt(2)
               elseif i < L+1 && j < L+1
                   im/sqrt(2)
               elseif i < L+1 && j > L+1
                   (-1)^(i-L)/sqrt(2)*im
               elseif i > L+1 && j < L+1
                   1/sqrt(2)
               end
           end
       end
   end
   return sparse(AA)
end

function test_r2c_y(L, cY, rY)
   Ts = BlockDiagonal([ ctran3(l)' for l = 0:L ]) |> sparse
   return cY ≈ Ts * rY
end

##

maxL = 20
cSH = CYlmBasis(maxL)
rSH = SCYlmBasis(maxL)

for nsamples = 1:30
   local R 
   R = rand_sphere()
   cY = evaluate(cSH, R)
   rY = evaluate(rSH, R)
   print_tf(@test test_r2c_y(maxL, cY, rY))
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

rSH = SCYlmBasis(10)
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


# ## -- check the laplacian implementation 

# using LinearAlgebra: tr
# using ForwardDiff
# P4 = Polynomials4ML

# function fwdΔ1(rYlm, x)
#    Y = evaluate(rYlm, x)
#    nY = length(Y)
#    _j(x) = ForwardDiff.jacobian(x -> evaluate(rYlm, x), x)[:]
#    _h(x) = reshape(ForwardDiff.jacobian(_j, x), (nY, 3, 3))
#    H = _h(x)
#    return [ tr(H[i, :, :]) for i = 1:nY ]
# end

# for x in X 
#    ΔY = P4.laplacian(rSH, x)
#    ΔYfwd = fwdΔ1(rSH, x)
#    print_tf(@test ΔYfwd ≈ ΔY)
# end
# println() 

# @info("check batched laplacian")
# ΔY1 = P4.laplacian(rSH, X)
# ΔY2 = similar(ΔY1)
# for (i, x) in enumerate(X)
#    ΔY2[i, :] = P4.laplacian(rSH, x)
# end
# println_slim(@test ΔY1 ≈ ΔY2)


# @info("check eval_grad_laplace")
# Y1, dY1, ΔY1 = P4.eval_grad_laplace(rSH, X)
# Y2, dY2 = evaluate_ed(rSH, X)
# ΔY2 = P4.laplacian(rSH, X)
# println_slim(@test Y1 ≈ Y2)
# println_slim(@test dY1 ≈ dY2)
# println_slim(@test ΔY1 ≈ ΔY2)

# using Zygote
# @info("Test rrule")
# using LinearAlgebra: dot 
# rSH = SCYlmBasis(10)

# for ntest = 1:30
#     local X
#     local Y
#     local Rnl
#     local u
    
#     X = [ rand_sphere() for i = 1:21 ]
#     Y = [ rand_sphere() for i = 1:21 ]
#     _x(t) = X + t * Y
#     A = evaluate(rSH, X)
#     u = randn(size(A))
#     F(t) = dot(u, evaluate(rSH, _x(t)))
#     dF(t) = begin
#         val, pb = Zygote.pullback(rSH, _x(t)) # TODO: write a pullback??
#         ∂BB = pb(u)[1] # pb(u)[1] returns NoTangent() for basis argument
#         return sum( dot(∂BB[i], Y[i]) for i = 1:length(Y) )
#     end
#     print_tf(@test fdtest(F, dF, 0.0; verbose = false))
# end
# println()

# ## Debugging code
# X = [ rand_sphere() for i = 1:21 ]
# Y = [ rand_sphere() for i = 1:21 ]
# _x(t) = X + t * Y
# A = evaluate(rSH, X)
# u = randn(size(A))
# F(t) = dot(u, evaluate(rSH, _x(t)))
# t = 1
# val, pb = Zygote.pullback(rSH, _x(t))
# pb
# ∂BB = pb(A)[1]

# dF(t) = begin
#       val, pb = Zygote.pullback(rSH, _x(t))
#       ∂BB = pb(u)[1] # pb(u)[1] returns NoTangent() for basis argument
#       return sum( dot(∂BB[i], Y[i]) for i = 1:length(Y) )
#    end
# fdtest(F, dF, 0.0; verbose = false)

# print_tf(@test fdtest(F, dF, 0.0; verbose = false))
