using LinearAlgebra, StaticArrays, Test, Printf, SparseArrays
using Polynomials4ML, Polynomials4ML.Testing
using Polynomials4ML: evaluate, evaluate_d, evaluate_ed 
using Polynomials4ML.Testing: print_tf, println_slim 
using ACEbase.Testing: fdtest
using HyperDualNumbers: Hyper
import SpheriCart
using SpheriCart: idx2lm, lm2idx

##                  

"""
This implements the original P4ML / ACE complex spherical harmonics basis 
up to L = 3. The convention is to L2-normalize on the sphere. The sign 
convention is not clear to me. It should be documented and clarified. 
"""
function explicit_shs(θ, φ)
   Y00 = 0.5 * sqrt(1/π)
   Y1m1 = 0.5 * sqrt(3/(2*π))*sin(θ)*exp(-im*φ)
   Y10 = 0.5 * sqrt(3/π)*cos(θ)
   Y11 = -0.5 * sqrt(3/(2*π))*sin(θ)*exp(im*φ)
   Y2m2 = 0.25 * sqrt(15/(2*π))*sin(θ)^2*exp(-2*im*φ)
   Y2m1 = 0.5 * sqrt(15/(2*π))*sin(θ)*cos(θ)*exp(-im*φ)
   Y20 = 0.25 * sqrt(5/π)*(3*cos(θ)^2 - 1)
   Y21 = -0.5 * sqrt(15/(2*π))*sin(θ)*cos(θ)*exp(im*φ)
   Y22 = 0.25 * sqrt(15/(2*π))*sin(θ)^2*exp(2*im*φ)
   Y3m3 = 1/8 * exp(-3 * im * φ) * sqrt(35/π) * sin(θ)^3
   Y3m2 = 1/4 * exp(-2 * im * φ) * sqrt(105/(2*π)) * cos(θ) * sin(θ)^2
   Y3m1 = 1/8 * exp(-im * φ) * sqrt(21/π) * (-1 + 5 * cos(θ)^2) * sin(θ)
   Y30 = 1/4 * sqrt(7/π) * (-3 * cos(θ) + 5 * cos(θ)^3)
   Y31 = -(1/8) * exp(im * φ) * sqrt(21/π) * (-1 + 5 * cos(θ)^2) * sin(θ)
   Y32 = 1/4 * exp(2 * im * φ) * sqrt(105/(2*π)) * cos(θ) * sin(θ)^2
   Y33 = -(1/8) * exp(3 * im * φ) * sqrt(35/π) * sin(θ)^3
   return [Y00, Y1m1, Y10, Y11, Y2m2, Y2m1, Y20, Y21, Y22,
         Y3m3, Y3m2, Y3m1, Y30, Y31, Y32, Y33]
end

lmax(::SphericalHarmonics{LMAX}) where {LMAX} = LMAX
lmax(::SolidHarmonics{LMAX}) where {LMAX} = LMAX

 
function eval_cY(rbasis, 𝐫)
   Yr = rbasis(𝐫)
   Yc = zeros(Complex{eltype(Yr)}, length(Yr))
   LMAX = lmax(rbasis)
   for l = 0:LMAX
      # m = 0 
      i_l0 = SpheriCart.lm2idx(l, 0)
      Yc[i_l0] = Yr[i_l0]
      # m ≠ 0 
      for m = 1:l 
         i_lm⁺ = SpheriCart.lm2idx(l,  m)
         i_lm⁻ = SpheriCart.lm2idx(l, -m)
         Ylm⁺ = Yr[i_lm⁺]
         Ylm⁻ = Yr[i_lm⁻]
         Yc[i_lm⁺] = (-1)^m * (Ylm⁺ + im * Ylm⁻) / sqrt(2)
         Yc[i_lm⁻] = (Ylm⁺ - im * Ylm⁻) / sqrt(2)
      end
   end 
   return Yc
end

function rand_angles() 
   θ = rand() * π
   φ = (rand()-0.5) * 2*π
   𝐫 = SVector(sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ))
   return 𝐫, θ, φ
end 

rand_sphere() = ( u = (@SVector randn(3)); u ./ norm(u) )

##

@info("Check a few conventions of the sphericart implementation")

# this test confirms that the above reference implementation of 
# sphericart -> complex spherical harmonics is consistent with 
# out old implementation.

r_spher = SphericalHarmonics(3)
r_solid = SolidHarmonics(3)

for ntest = 1:30 
   𝐫, θ, φ = rand_angles() 
   Yr = r_spher(𝐫)
   Zr = r_solid(𝐫)
   Yref = explicit_shs(θ, φ)
   Yc = eval_cY(r_spher, 𝐫)
   print_tf(@test Yr ≈ Zr)
   print_tf(@test Yc ≈ Yref)
end 
println() 

##

@info("Confirm L2-orthonormalization")
R = [ rand_sphere() for _ = 1:1_000_000 ]
Y = SpheriCart.compute(r_spher, R)
G = (Y' * Y) * 4 * π / length(R)
println_slim(@test norm(G - I, Inf) < 0.1)

##

@info("Check consistency of spherical and solid")
for ntest = 1:30 
   𝐫̂ = rand_sphere() 
   r = 2 * rand()
   𝐫 = r * 𝐫̂
   Yr = Vector(r_spher(𝐫))
   Zr = r_solid(𝐫)
   for l = 0:3, m = -l:l
      i = SpheriCart.lm2idx(l, m)
      Yr[i] *= r^l
   end
   print_tf(@test Yr ≈ Zr)
end


#=
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

using Zygote
@info("Test rrule")
using LinearAlgebra: dot 
rSH = SCYlmBasis(10)

for ntest = 1:30
    local X
    local Y
    local Rnl
    local u
    
    X = [ rand_sphere() for i = 1:21 ]
    Y = [ rand_sphere() for i = 1:21 ]
    _x(t) = X + t * Y
    A = evaluate(rSH, X)
    u = randn(size(A))
    F(t) = dot(u, evaluate(rSH, _x(t)))
    dF(t) = begin
        val, pb = Zygote.pullback(rSH, _x(t)) # TODO: write a pullback??
        ∂BB = pb(u)[1] # pb(u)[1] returns NoTangent() for basis argument
        return sum( dot(∂BB[i], Y[i]) for i = 1:length(Y) )
    end
    print_tf(@test fdtest(F, dF, 0.0; verbose = false))
end
println()

=#