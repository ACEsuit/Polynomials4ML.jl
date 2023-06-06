# --------------------- Jacobi Weights 
# this includes in particular Legendre, Gegenbauere, Chebyshev 

export chebyshev_basis, legendre_basis, jacobi_basis

using SpecialFunctions
using QuadGK

struct JacobiWeights{T}
   α::T
   β::T
   normalize::Bool 
end

chebyshev_weights(normalize=true) = JacobiWeights(-0.5, -0.5, normalize)

legendre_weights(normalize=true) = JacobiWeights(0.0, 0.0, normalize)


function orthpolybasis(N::Integer, W::JacobiWeights{T}) where {T} 
   α = W.α
   β = W.β
   A = zeros(T, N)
   B = zeros(T, N)
   C = zeros(T, N)
   A[1] = 1
   if N > 0 
      A[2] = (α+β+2)/2 
      B[2] = - (α+β+2)/2 + (α+1)
   end 
   for n = 2:N-1
      c1 = big(2*n*(n+α+β)*(2*n+α+β-2))  
      c2 = big(2*n+α+β-1)                
      A[n+1] = T( big(2*n+α+β)*big(2*n+α+β-2)*c2 / c1 )    
      B[n+1] = T( big(α^2 - β^2) * c2 / c1 )               
      C[n+1] = T( big(-2*(n+α-1)*(n+β-1)*(2n+α+β)) / c1 )  
   end
   basis = OrthPolyBasis1D3T(A, B, C)
   basis.meta["weights"] = W
   if W.normalize
      integrand = x -> evaluate(basis, x).^2 * ((1-x)^α * (1+x)^β)
      g = sqrt.(quadgk(integrand, -1.0+1e-15, 1.0-1e-15; atol=1e-10)[1])

      # new implementation - incorrect
      # g0 = sqrt(quadgk(x -> (1-x)^α * (1+x)^β, -1.0+1e-15, 1.0-1e-15; atol=1e-12)[1])
      # nrm_jacobi(n) = (n == 0) ? g0 : T( big(2^(α+β+1))  * 
      #                         big(gamma(n+α+1)) * big(gamma(n+β+1)) / big(gamma(n+α+β+1))
      #                         / big(2*n+α+β+1) / big(factorial(n)) )
      # g = [ nrm_jacobi(n) for n = 0:N-1 ]
      # display([g g1])

      basis.A[1] /= g[1] 
      basis.A[2] /= g[2] 
      basis.B[2] /= g[2] 
      for n = 3:N 
         basis.A[n] *= g[n-1]/g[n] 
         basis.B[n] *= g[n-1]/g[n] 
         basis.C[n] *= g[n-2]/g[n] 
      end
   end
   return basis 
end

"""
`chebyshev_basis(N::Integer)`: 

Constructs an `OrthPolyBasis1D3T` object representing a possibly rescaled version of the basis of Chebyshev polynomials of the first kind. `N` is the length of the basis, not the degree. 

Careful: the normalisation may be non-standard. 
"""
function chebyshev_basis(N::Integer; normalize=false) 
   cheb = orthpolybasis(N, chebyshev_weights(normalize))
   if normalize 
      cheb.A[1] = sqrt(1/π)
      cheb.A[2] = sqrt(2/π)
      cheb.C[3] = - sqrt(2) 
      cheb.A[3:end] .= 2 
      cheb.B[:] .= 0 
      cheb.C[4:end] .= -1 
   else 
      cheb.A[1] = 1
      cheb.A[2] = 1
      cheb.A[3:end] .= 2 
      cheb.B[:] .= 0 
      cheb.C[3:end] .= -1 
   end
   return cheb 
end 
      

"""
`legendre_basis(N::Integer)`: 

Constructs an `OrthPolyBasis1D3T` object representing a possibly rescaled version of the basis of Legendre polynomials (L2 orthonormal on [-1, 1]). `N` is the length of the basis, not the degree. 

Careful: the normalisation may be non-standard. 
"""
legendre_basis(N::Integer; normalize=true) = 
      orthpolybasis(N, legendre_weights(normalize))

"""
`jacobi_basis(N::Integer, α::Real, β::Real)`: 

Constructs an `OrthPolyBasis1D3T` object representing a possibly rescaled version of the basis of Jacobi polynomials `Jαβ`. `N` is the length of the basis, not the degree. 

Careful: the normalisation may be non-standard. 
"""
jacobi_basis(N::Integer, α::Real, β::Real; normalize=true) =
       orthpolybasis(N, JacobiWeights(α, β, normalize))