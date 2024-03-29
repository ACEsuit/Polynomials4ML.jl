
using Polynomials4ML, Test
using Polynomials4ML: evaluate, evaluate_d, evaluate_dd
using Polynomials4ML.Testing: println_slim, test_derivatives, print_tf
using LinearAlgebra: I, norm, dot 
using QuadGK
using ACEbase.Testing: fdtest
using Printf
using ChainRulesCore: rrule 
using Zygote

@info("Testing OrthPolyBasis1D3T")


##

for ntest = 1:3
   @info("Test a randomly generated polynomial basis - $ntest")
   N = rand(5:15)
   basis = OrthPolyBasis1D3T(randn(N), randn(N), randn(N))
   test_derivatives(basis, () -> rand())
end

##

@info("Test the Legendre polynomials")
@info("    orthogonality")
N = 20 
legendre = legendre_basis(N, normalize=true)
G = quadgk(x -> legendre(x) * legendre(x)', -1, 1)[1]
println_slim(@test round.(G, digits=6) ≈ I)
@info("    derivatives")
test_derivatives(legendre, () -> 2*rand()-1)

##

for ntest = 1:3
   local N, G
   α = 1 + rand() 
   β = 1 + rand() 
   @info("Test the Random Jacobi Polynomials")
   @info("   α = $α, β = $β")
   @info("    orthogonality")
   N = 20 
   jacobi = jacobi_basis(N, α, β, normalize=true)
   G = quadgk(x -> (1-x)^α * (x+1)^β * jacobi(x) * jacobi(x)', -1, 1)[1]
   println_slim(@test round.(G, digits=6) ≈ I)
   @info("    derivatives")
   test_derivatives(legendre, () -> 2*rand()-1)
end 

##


@info("Test normalized cheb basis") 
@info("   coeffs")
cheb = chebyshev_basis(N, normalize=true)
println_slim(@test all([ 
   cheb.A[1] ≈ sqrt(1/π), 
   cheb.A[2] ≈ sqrt(2/π), 
   cheb.C[3] ≈ -sqrt(2), 
   norm(cheb.A[3:end] .- 2, Inf) < 1e-12, 
   norm(cheb.B, Inf) == 0, 
   norm(cheb.C[4:end] .+ 1) < 1e-12, ] ))
@info("   orthogonality")
G = quadgk(x -> (1-x)^(-0.5) * (x+1)^(-0.5) * cheb(x) * cheb(x)', -1, 1)[1]
println_slim(@test round.(G, digits=6) ≈ I)
@info("     derivatives")
test_derivatives(cheb, () -> 2*rand()-1)


@info("Check correctness of Chebyshev Basis (normalize=false)")
cheb = chebyshev_basis(N, normalize=false)
@info("     recursion coefficients")
println_slim(@test all([ 
         cheb.A[1] ≈ 1, 
         all(cheb.B[:] .== 0), 
         cheb.A[2] ≈ 1, 
         all(cheb.A[3:end] .≈ 2), 
         all(cheb.C[3:end] .≈ -1), ]))

@info("    consistency with ChebBasis")
cheb2 = ChebBasis(N)
println_slim(@test all( (x = 2*rand()-1; cheb(x) ≈ cheb2(x)) for _=1:30 ))

##
@info("Testing rrule")

using LinearAlgebra: dot 
N = 10
for ntest = 1:30
   bBB = randn(N)
   bUU = randn(N)
   _BB(t) = bBB + t * bUU
   bA2 = cheb(bBB)
   u = randn(size(bA2))
   F(t) = dot(u, Polynomials4ML.evaluate(cheb, _BB(t)))
   dF(t) = begin
       val, pb = Zygote.pullback(evaluate, cheb, _BB(t))
       ∂BB = pb(u)[2] # pb(u)[1] returns NoTangent() for basis argument
       return sum( dot(∂BB[i], bUU[i]) for i = 1:length(bUU) )
   end
   print_tf(@test fdtest(F, dF, 0.0; verbose = false))
end
println()

##
# ---------------- Double Pullback Test ----------------
using ForwardDiff
P4ML = Polynomials4ML

@info("Testing double-pullback")

Nx = 6; Np = 11
cheb = chebyshev_basis(Np, normalize=false)
X = 2*rand(Nx) .- 1

@info("    first double-check first pullback a different way")
bP1 = cheb(X)
val, pb = rrule(evaluate, cheb, X)
println_slim(@test val ≈ bP1)
dP1 = P4ML.evaluate_d(cheb, X)
u = randn(size(bP1))
println_slim(@test pb(u)[3] ≈  [ dot(u[j, :], collect(dP1[j, :])) for j = 1:Nx ])

##

@info("    now check the second-order pullback")

function F(u, X) 
   d = 1:length(X) 
   val, pb = rrule(evaluate, cheb, X)
   return dot(d, pb(u)[3])
end 

g_uX = Zygote.gradient(F, u, X)

gf_u = ForwardDiff.gradient(_u -> F(_u, X), u)
println_slim(@test (gf_u ≈ g_uX[1]))

gf_X = ForwardDiff.gradient(_X -> F(u, _X), X)
println_slim(@test (gf_X ≈ g_uX[2]))
