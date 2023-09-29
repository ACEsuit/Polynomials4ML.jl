using Polynomials4ML: legendre_basis, RYlmBasis, ScalarPoly4MLBasis, PooledSparseProduct, OrthPolyBasis1D3T, PooledEmbeddings
using StaticArrays, LinearAlgebra
using ObjectPools: acquire!, release!
using Polynomials4ML
using ACEbase.Testing: fdtest, print_tf
using Test
using Printf
using Random
using Lux
using Zygote

function _generate_basis(; order=2, len = 50)
   NN = [ rand(5:10) for _ = 1:order ]
   spec = sort([ ntuple(t -> rand(1:NN[t]), order) for _ = 1:len])
   return PooledSparseProduct(spec)
end

Rnl = legendre_basis(10)
Ylm = RYlmBasis(10)

pooling = _generate_basis()

X = [ @SVector(randn(3)) for i in 1:3 ]

embeddings = (Rnl, Ylm)
embed_and_pool = Polynomials4ML.PooledEmbeddings(embeddings, pooling)


@info("Test evaluate")
for ntest = 1:30
   bX = [ @SVector(randn(3)) for i in 1:3 ]
   out = evaluate(embed_and_pool, X)
   out_Rnl, out_Ylm = Rnl(X), Ylm(X)
   out_pooling = pooling((out_Rnl, out_Ylm))
   print_tf(@test out ≈ out_pooling)
end

@info("Test rrule")
for ntest = 1:20
   local bX, bu, u, bA
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


@info("Testing consistency with lux layer")
l_embed_and_pool = Polynomials4ML.lux(embed_and_pool)
ps, st = Lux.setup(MersenneTwister(1234), l_embed_and_pool)
@info("Test evaluate")
for ntest = 1:30
   bX = [ @SVector(randn(3)) for i in 1:3 ]
   print_tf(@test l_embed_and_pool(X, ps, st)[1] ≈ embed_and_pool(X))
end


@info("Testing rrule working correectly")
for ntest = 1:30
   local bX, bu, u
   bX = [ @SVector(randn(3)) for i in 1:3 ]
   bu = [ @SVector(randn(3)) for i in 1:3 ]
   _BB(t) = bX + t * bu
   bA = l_embed_and_pool(X, ps, st)[1]
   u = randn(size(bA))
   F(t) = dot(u,  l_embed_and_pool(_BB(t), ps, st)[1])
   dF(t) = begin
      out, pb = Zygote.pullback(evaluate, embed_and_pool, _BB(t))
      ∂BB = pb(u)[2]
      return sum( dot(∂BB[i], bu[i]) for i = 1:length(bX) )
   end
   print_tf(@test fdtest(F, dF, 0.0; verbose = false))
end
