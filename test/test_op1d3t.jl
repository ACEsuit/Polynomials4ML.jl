
using Polynomials4ML, Test
using Polynomials4ML: _generate_input
using Polynomials4ML.Testing: println_slim, print_tf, test_all 
using LinearAlgebra: I, norm, dot 
using QuadGK


@info("Testing OrthPolyBasis1D3T")

##

for ntest = 1:3
   @info("Test a randomly generated polynomial basis - $ntest")
   N = rand(5:15)
   basis = OrthPolyBasis1D3T(randn(N), randn(N), randn(N))
   test_all(basis)
end

##

@info("Test the Legendre polynomials")
@info("    orthogonality")
N = 20 
legendre = legendre_basis(N, normalize=true)
G = quadgk(x -> legendre(x) * legendre(x)', -1, 1)[1]
println_slim(@test round.(G, digits=6) ≈ I)
@info("    derivatives")
test_all(legendre)

##

for ntest = 1:3
   local N, G
   α = 1 + rand() 
   β = 1 + rand() 
   @info("Test Random Jacobi Polynomials")
   @info("   α = $α, β = $β")
   @info("    orthogonality")
   N = 20 
   jacobi = jacobi_basis(N, α, β, normalize=true)
   G = quadgk(x -> (1-x)^α * (x+1)^β * jacobi(x) * jacobi(x)', -1, 1)[1]
   println_slim(@test round.(G, digits=6) ≈ I)
   @info("    derivatives, etc")
   test_all(jacobi)
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
@info("     derivatives, etc")
test_all(cheb)

## 

@info("Test chebyshev_basis vs ChebBasis{N}")

basis1 = ChebBasis(N)
basis2 = chebyshev_basis(N; normalize=false)
basis3 = chebyshev_basis(N; normalize=true)
x = _generate_input(basis1)
r13 = basis1(x) ./ basis3(x)
for _ = 1:10 
   local x 
   x = _generate_input(basis1)
   P1 = basis1(x)
   P2 = basis2(x)
   P3 = basis3(x)
   print_tf(@test P1 ≈ P2)
   print_tf(@test P1 ≈ P3 .* r13)
end 
println() 

##

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
