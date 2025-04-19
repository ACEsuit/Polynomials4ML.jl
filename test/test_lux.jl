using Polynomials4ML, Test, StaticArrays, LuxCore, Zygote, ForwardDiff
using Polynomials4ML: lux, _generate_input
using Random: default_rng
using ACEbase.Testing: println_slim, print_tf
using LinearAlgebra: dot, I 
rng = default_rng()

##

@info("Testing Lux Layers for basic basis sets")

test_bases = [ chebyshev_basis(10), 
               ChebBasis(8), 
               RTrigBasis(10), 
               CTrigBasis(10), 
               MonoBasis(10), 
               legendre_basis(10),
               real_sphericalharmonics(5), 
               real_solidharmonics(5), 
               complex_sphericalharmonics(5), 
               complex_solidharmonics(5), ]

##

for basis in test_bases
   @info("Lux layer test for $(typeof(basis).name.name)")
   local B1, B2, x
   local ps, st
   nX = rand(8:16)
   X = [ _generate_input(basis) for _ = 1:nX ]
   B1 = basis(X)
   l = lux(basis)
   ps, st = LuxCore.setup(rng, l)
   B2, _ = l(X, ps, st)
   println_slim(@test B1 == B2)

   if !( eltype(eltype(B1)) <: Real)
      println("basis is complex; skipping AD test")
      continue 
   end

   val1, pb1 = Zygote.pullback(LuxCore.apply, l, X, ps, st)
   val2, pb2 = Zygote.pullback(Polynomials4ML.evaluate, basis, X)
   Δ = randn(eltype(val2), size(val2))
   ∂1 = pb1( (Δ, NamedTuple()) )
   ∂2 = pb2(Δ)

   function _fd_pb(basis, X::AbstractVector{<: Number}, Δ)
      return ForwardDiff.gradient(_X -> real(sum(basis(_X) .* Δ)), X) 
   end

   function _fd_pb(basis, X::AbstractVector{SVector{N, T}}, Δ) where {N, T}
      _F(Xv) = dot(basis(reinterpret(SVector{N, eltype(Xv)}, Xv)), Δ)
      Xvec = collect( reinterpret(T, X) )
      Gvec = ForwardDiff.gradient(_F, Xvec)
      return collect(reinterpret(SVector{N, T}, Gvec))
   end

   ∂_ad = _fd_pb(basis, X, Δ)

   println_slim(@test val1[1] == val2)
   println_slim(@test ∂1[2] ≈ ∂2[2] ≈ ∂_ad) 
end 


