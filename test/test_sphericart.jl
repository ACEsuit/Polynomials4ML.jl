using LinearAlgebra, StaticArrays, Test, Printf, SparseArrays, 
      ForwardDiff, Random 
using Polynomials4ML, Polynomials4ML.Testing
using Polynomials4ML: evaluate, evaluate_d, evaluate_ed, 
                      idx2lm, lm2idx, maxl
using Polynomials4ML.Testing: print_tf, println_slim, 
                     test_evaluate_xx, test_withalloc, 
                     test_chainrules, test_ka_evaluate
using ACEbase.Testing: fdtest
import SpheriCart

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

# real solid harmonics 
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
   return [Y00, 
            Y1m1, Y10, Y11, 
            Y2m2, Y2m1, Y20, Y21, Y22,
            Y3m3, Y3m2, Y3m1, Y30, Y31, Y32, Y33, 
            Y4m4, Y4m3, Y4m2, Y4m1, Y40, Y41, Y42, Y43, Y44]
end

function eval_cY_from_rY(rbasis, 𝐫)
   Yr = rbasis(𝐫)
   Yc = zeros(Complex{eltype(Yr)}, length(Yr))
   LMAX = maxl(rbasis)
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
# our old implementation.

r_spher = real_sphericalharmonics(3)  # SphericalHarmonics(3)
r_solid = real_solidharmonics(3)      # SolidHarmonics(3)
c_spher = complex_sphericalharmonics(3)  # SphericalHarmonics(3)
c_solid = complex_solidharmonics(3)      # SolidHarmonics(3)


for ntest = 1:30 
   local 𝐫, θ, φ
   𝐫, θ, φ = rand_angles() 
   Yr = r_spher(𝐫)
   Zr = r_solid(𝐫)
   Yref = explicit_shs(θ, φ)
   Yc1 = eval_cY_from_rY(r_spher, 𝐫)
   Yc2 = c_spher(𝐫)
   Zc = c_solid(𝐫)
   print_tf(@test Yr ≈ Zr)
   print_tf(@test Yc1 ≈ Yref ≈ Yc2 ≈ Zc)
end 
println() 

##

@info("Test Racah normalization")
racah = real_solidharmonics(4; normalisation = :racah)
for ntest = 1:20 
   𝐫 = @SVector randn(3) 
   Z1 = racah(𝐫)
   Z2 = explicit_rsh(𝐫)
   print_tf(@test Z1 ≈ Z2) 
end
println() 

##

@info("Confirm L2-orthonormalization - real")
Random.seed!(1)
R = [ rand_sphere() for _ = 1:1_000_000 ]
Y = r_spher(R)
G = (Y' * Y) * 4 * π / length(R)
# @show norm(G - I, Inf)
println_slim(@test norm(G - I, Inf) < 0.01)

@info("Confirm L2-orthonormalization - complex")
Y = c_spher(R)
G = (Y' * Y) * 4 * π / length(R)
# @show norm(G - I, Inf)
println_slim(@test norm(G - I, Inf) < 0.01)

##

@info("Check consistency of spherical and solid")
for ntest = 1:30 
   local r, 𝐫 
   𝐫̂ = rand_sphere() 
   r = 2 * rand()
   𝐫 = r * 𝐫̂
   Yr = Vector(r_spher(𝐫))
   Zr = r_solid(𝐫)
   Yc = Vector(c_spher(𝐫))
   Zc = c_solid(𝐫)
   for l = 0:3, m = -l:l
      i = SpheriCart.lm2idx(l, m)
      Yr[i] *= r^l
      Yc[i] *= r^l
   end
   print_tf(@test Yr ≈ Zr)
   print_tf(@test Yc ≈ Zc)
end
println() 

##

@info("Test: check derivatives of 3D harmonics")
bases = [ real_sphericalharmonics(10), 
          real_solidharmonics(11), 
          real_solidharmonics(12, normalisation = :racah), 
          complex_sphericalharmonics(9), 
          complex_solidharmonics(11) ]

for basis in bases
   @info("Tests for $(basis)")
   test_chainrules(basis)
   test_evaluate_xx(basis)
   test_withalloc(basis)
end

# testing the KA evaluation; this is currently restricted to just 
# the real solid harmonics; this needs to be fixed at the SpheriCart end 
for basis in bases[2:3]
   test_ka_evaluate(basis)
end
