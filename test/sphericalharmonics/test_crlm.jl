using LinearAlgebra, StaticArrays, Test, Printf
using Polynomials4ML, Polynomials4ML.Testing
using Polynomials4ML: SphericalCoords, 
                      dspher_to_dcart, cart2spher, spher2cart, index_y
using Polynomials4ML: evaluate, evaluate_d, evaluate_ed 
using Polynomials4ML.Testing: print_tf, println_slim 
using ACEbase.Testing: fdtest

verbose = false

"""
γₗᵐ(x, y, z) = Nₗₘ[x+sgn(m) iy]ᵃᵇˢ⁽ᵐ⁾∑ₜ₌₀ᶠˡᵒᵒʳ⁽⁰⁵⁽ˡ⁻ᵃᵇˢ⁽ᵐ⁾⁾⁾ Cₜˡᵃᵇˢ⁽ᵐ⁾(x²+y²)ᵗzˡ⁻²ᵗ⁻ᵃᵇˢ⁽ᵐ⁾
Nₗₘ = (-1)⁽⁰⁵⁽ᵐ⁺ᵃᵇˢ⁽ᵐ⁾⁾1/(2ᵃᵇˢ⁽ᵐ⁾l!) √(2l+1)/4π(l+|m|)!(l-|m|)!
Cₜˡᵃᵇˢ⁽ᵐ⁾ = (-1/4)ᵗ(l-t,|m|+t)(l,t)
"""
function explicit_solid_harmonics(l::Int,m::Int,X::AbstractVector)
    function cumprod2(l)
        if l == 0
            return 1
        else
            return cumprod(collect(1:l))[end]
        end
    end 
    x, y, z = X
    Nlm = (-1)^((m+abs(m))/2) * 1/(2^(abs(m))*cumprod2(l)) * sqrt((2l+1)/(4*pi) * cumprod2(l+abs(m)) * cumprod2(l-abs(m)))
    a = 0
    for t = 0:Int(floor((l-abs(m))/2))
        Ctlm = (-1/4)^t * binomial(l-t,abs(m)+t) * binomial(l,t)
        a += Nlm * (x+sign(m)*im * y)^(abs(m)) * Ctlm * (x^2+y^2)^t * z^(l-2*t-abs(m))
    end
    return a
end
##

@info("Test: check complex solid harmonics against explicit expressions")
nsamples = 30
for n = 1:nsamples
    local X
    l = rand(collect(1:10))
    m = rand(collect(1:l))
    θ = rand() * π
    φ = (rand()-0.5) * 2*π
    r = 0.1+rand()
    X = SVector(r*sin(θ)*cos(φ), r*sin(θ)*sin(φ), r*cos(θ))
    SH = CRlmBasis(l)
    Y = evaluate(SH, X)[index_y(l,m)]
    Yex = explicit_solid_harmonics(l, m, X)
    print_tf((@test Y ≈ Yex))
end
println()

##
@info("      ... same near pole")
nsamples = 30
for n = 1:nsamples
    local X
    l = rand(collect(1:10))
    m = rand(collect(1:l))
    θ = rand() * 1e-9
    if θ < 1e-10
        θ = 0.0
    end
    φ = (rand()-0.5) * 2*π
    r = 0.1+rand()
    X = SVector(r*sin(θ)*cos(φ), r*sin(θ)*sin(φ), r*cos(θ))
    SH = CRlmBasis(l)
    Y = evaluate(SH, X)[index_y(l,m)]
    Yex = explicit_solid_harmonics(l, m, X)
    print_tf((@test Y ≈ Yex || norm(Y - Yex, Inf) < 1e-12))
end
println()

@info("Test: check complex solid harmonics against spherical harmonics times r^l")
nsamples = 30
for n = 1:nsamples
    local X
    local Y2
    l = rand(collect(1:10))
    m = rand(collect(1:l))
    θ = rand() * π
    φ = (rand()-0.5) * 2*π
    r = 0.1+rand()
    X = SVector(r*sin(θ)*cos(φ), r*sin(θ)*sin(φ), r*cos(θ))
    SH = CRlmBasis(l)
    SH2 = CYlmBasis(l)
    Y = evaluate(SH, X)[index_y(l,m)]
    Y2 = evaluate(SH2, X)[index_y(l,m)] * r^l
    print_tf((@test Y ≈ Y2))
end
println()

##
@info("      ... same near pole")
nsamples = 30
for n = 1:nsamples
    local X
    local Y2
    l = rand(collect(1:10))
    m = rand(collect(1:l))
    θ = rand() * 1e-9
    if θ < 1e-10
        θ = 0.0
    end
    φ = (rand()-0.5) * 2*π
    r = 0.1+rand()
    X = SVector(r*sin(θ)*cos(φ), r*sin(θ)*sin(φ), r*cos(θ))
    SH = CRlmBasis(l)
    SH2 = CYlmBasis(l)
    Y = evaluate(SH, X)[index_y(l,m)]
    Y2 = evaluate(SH2, X)[index_y(l,m)] * r^l
    print_tf((@test Y ≈ Y2 || norm(Y - Y2, Inf) < 1e-12))
end
println()


##
using Polynomials4ML: SphericalCoords, ALPolynomials
verbose=false
@info("Test: check derivatives of complex solid harmonics")
for nsamples = 1:30
   local X
   local Y1
   X = @SVector rand(3)
   SH = CRlmBasis(5)
   Y1 = evaluate(SH, X)
   Y, dY = evaluate_ed(SH, X)
   print_tf(@test(Y ≈ Y1))
   DY = Matrix(transpose(hcat(dY...)))
   errs = []
   verbose && @printf("     h    | error \n")
   for p = 2:10
      local h = 0.1^p
      DYh = similar(DY)
      Rh = Vector(X)
      for i = 1:3
         Rh[i] += h
         DYh[:, i] = (evaluate(SH, SVector(Rh...)) - Y) / h
         Rh[i] -= h
      end
      push!(errs, norm(DY - DYh, Inf))
      verbose && @printf(" %.2e | %.2e \n", h, errs[end])
   end
   success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
   print_tf(@test success)
end
println()



##

@info("Check consistency of standard and batched ALP evaluation ")
using Polynomials4ML: rand_sphere
basis = CRlmBasis(5)
R = [ rand_sphere() for _ = 1:32 ] 

S = cart2spher.(R)
P1, dP1 = Polynomials4ML.evaluate_ed(basis.alp, S)
P2 = copy(P1) 
dP2 = copy(dP1)

for i = 1:length(R)
   P2[i, :], dP2[i, :] = Polynomials4ML.evaluate_ed(basis.alp, S[i])
end

println_slim(@test P2 ≈ P1)
println_slim(@test dP2 ≈ dP2)

## 

@info("Check consistency of standard and batched Rlm evaluation ")

Yb = evaluate(basis, R)
Yb1, dYb1 = evaluate_ed(basis, R)

Ys = copy(Yb)
Ys2 = copy(Ys)
dYs2 = copy(dYb1)

for i = 1:length(R)
   Ys[i, :] = evaluate(basis, R[i])
   Ys2[i, :], dYs2[i, :] = evaluate_ed(basis, R[i])
end

println_slim(@test Yb ≈ Ys ≈ Ys2 ≈ Yb1) 
println_slim(@test dYb1 ≈ dYs2)

##

using Zygote
@info("Test rrule")
using LinearAlgebra: dot 
basis = CRlmBasis(5)
for ntest = 1:30
   local X
   local Y
   local u
    
   X = [ rand_sphere() for i = 1:21 ]
   Y = [ rand_sphere() for i = 1:21 ]
   _x(t) = X + t * Y
   A = evaluate(basis, X)
   u = randn(size(A))
   F(t) = dot(u, evaluate(basis, _x(t)))
   dF(t) = begin
       val, pb = Zygote.pullback(basis, _x(t))
       ∂BB = pb(u)[1] # pb(u)[1] returns NoTangent() for basis argument
       return sum( dot(∂BB[i], Y[i]) for i = 1:length(Y) )
   end
   print_tf(@test fdtest(F, dF, 0.0; verbose = true))
#end