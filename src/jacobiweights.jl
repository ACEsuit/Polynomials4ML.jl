# --------------------- Jacobi Weights 
# this includes in particular Legendre, Gegenbauere, Chebyshev 

export chebyshev_basis, legendre_basis, jacobi_basis

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
   meta = Dict{String, Any}("weights" => W)
   basis = OrthPolyBasis1D3T(A, B, C, meta)
   if W.normalize
      integrand = x -> evaluate(basis, x).^2 * ((1-x)^α * (1+x)^β)
      g = sqrt.(quadgk(integrand, -1.0, 1.0)[1])
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


chebyshev_basis(N::Integer; normalize=true) = 
      orthpolybasis(N, chebyshev_weights(normalize))

legendre_basis(N::Integer; normalize=true) = 
      orthpolybasis(N, legendre_weights(normalize))

jacobi_basis(N::Integer, α::Real, β::Real; normalize=true) =
       orthpolybasis(N, JacobiWeights(α, β, normalize))