
using Polynomials4ML, Test
using Polynomials4ML: evaluate, evaluate_d, evaluate_dd
using Polynomials4ML.Testing: println_slim, test_derivatives
using LinearAlgebra: I
using QuadGK

@info("Testing OrthPolyBasis1D3T")


##

for ntest = 1:3
   @info("Test a randomly generated polynomial basis - $ntest")
   N = rand(10:15)
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

@info("Check correctness of Chebyshev Basis")
cheb = chebyshev_basis(N, normalize=true)
@info("     recursion coefficients")
println_slim(@test all([ 
         cheb.A[1] ≈ sqrt(1/π), 
         all(cheb.B[:] .== 0), 
         cheb.A[2] ≈ sqrt(2/π), 
         all(cheb.A[3:end] .≈ 2), 
         cheb.C[3] ≈ - sqrt(2), 
         all(cheb.C[4:end] .≈ -1), ]))         
@info("     derivatives")
test_derivatives(legendre, () -> 2*rand()-1)

