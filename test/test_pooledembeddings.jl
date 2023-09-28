using Polynomials4ML: legendre_basis, RYlmBasis, ScalarPoly4MLBasis, PooledSparseProduct, OrthPolyBasis1D3T, PooledEmebddings
using StaticArrays, LinearAlgebra
using ObjectPools: acquire!, release!
using Polynomials4ML
using ACEbase.Testing: fdtest, print_tf
using Test
using Printf

function _generate_basis(; order=2, len = 50)
   NN = [ rand(5:10) for _ = 1:order ]
   spec = sort([ ntuple(t -> rand(1:NN[t]), order) for _ = 1:len])
   return PooledSparseProduct(spec)
end

function Polynomials4ML.evaluate!(Y::AbstractArray, basis::OrthPolyBasis1D3T, X::Vector{SVector{3, T}}) where T
   nX = length(X)
   R = acquire!(basis.pool, :R, (nX,), T)
   @simd ivdep for i = 1:nX
      R[i] = norm(X[i])
   end    
   evaluate!(Y, basis, R)
   release!(R)
	return Y
end

using ChainRulesCore

function ChainRulesCore.rrule(::typeof(Polynomials4ML.evaluate), basis::OrthPolyBasis1D3T, X::Vector{SVector{3, T}}) where T
   R = norm.(X)
   P, dP = evaluate_ed(basis, R)

   function pb(∂)
      ∂X = zero(X)
      for n = 1:size(dP, 2)
         @simd ivdep for a = 1:length(X)
            ∂X[a] += ∂[a, n] * dP[a, n] * X[a] / R[a]
         end
      end
      return (NoTangent(), NoTangent(), ∂X)
   end

  return P, pb
end

Polynomials4ML._valtype(basis::OrthPolyBasis1D3T{T}, ::Type{<: StaticVector{3, S}}) where {T <: Real, S <: Real} = promote_type(T, S)

Rnl = legendre_basis(10)
Ylm = RYlmBasis(10)

pooling = _generate_basis()

X = [ @SVector(randn(3)) for i in 1:3 ]

embeddings = (Rnl, Ylm)
embed_and_pool = Polynomials4ML.PooledEmebddings(embeddings, pooling)


# @info("Test evaluate")
# for ntest = 1:30
#    bX = [ @SVector(randn(3)) for i in 1:3 ]
#    evaluate(embed_and_pool, X)
# end

@info("Test rrule")
using Zygote
for ntest = 1:30
   bX = [ @SVector(randn(3)) for i in 1:3 ]
   bu = [ @SVector(randn(3)) for i in 1:3 ]
   _BB(t) = bX + t * bu
   bA = evaluate(embed_and_pool, X)
   u = randn(size(bA))
   F(t) = dot(u, Polynomials4ML.evaluate(embed_and_pool, _BB(t)))
   dF(t) = begin
      out, pb = Zygote.pullback(evaluate, embed_and_pool, _BB(t))
      ∂BB = pb(u)[2]
      return sum( dot(∂BB[i], bu[i]) for i = 1:length(bX) )
   end
   print_tf(@test fdtest(F, dF, 0.0; verbose = false))
end