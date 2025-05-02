using Polynomials4ML, Test, StaticArrays, LuxCore, Zygote, ForwardDiff
using Polynomials4ML: lux, _generate_input
using Random: default_rng
using ACEbase.Testing: println_slim, print_tf
using LinearAlgebra: dot, I 
rng = default_rng()

function _fd_pb(basis, X::AbstractVector{<: Number}, Δ)
   return ForwardDiff.gradient(_X -> real(sum(basis(_X) .* Δ)), X) 
end

function _fd_pb(basis, X::AbstractVector{SVector{N, T}}, Δ) where {N, T}
   _F(Xv) = dot(basis(reinterpret(SVector{N, eltype(Xv)}, Xv)), Δ)
   Xvec = collect( reinterpret(T, X) )
   Gvec = ForwardDiff.gradient(_F, Xvec)
   return collect(reinterpret(SVector{N, T}, Gvec))
end


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
# basis = test_bases[8]
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

   # evaluate the basis and get the pullback operator  
   val1, pb1 = Zygote.pullback(LuxCore.apply, l, X, ps, st)
   val2, pb2 = Zygote.pullback(Polynomials4ML.evaluate, basis, X, ps, st)
   # evaluate the pullback on a random cotangent 
   Δ = randn(eltype(val2), size(val2))
   ∂1 = pb1( (Δ, NamedTuple()) )
   ∂2 = pb2(Δ)
   # compute the gradient again using ForwardDiff to compare 
   ∂_ad = _fd_pb(basis, X, Δ)
   # check that all three give the same result 
   println_slim(@test val1[1] ≈ val2)
   println_slim(@test ∂1[2] ≈ ∂2[2] ≈ ∂_ad) 

   # look at gradients with respect to the parameters 
   _foo = p -> dot(Δ, LuxCore.apply(l, X, p, st)[1])
   g = Zygote.gradient(_foo, ps)[1] 
   if sizeof(ps) == 0 
      println_slim(@test (isnothing(g) || isempty(g)))
   else
      pvec, _restruct = destructure(ps) 
      g2 = _restruct( ForwardDiff.gradient(p -> _foo(_restruct(p)), pvec) )
      println_slim(@test g ≈ g2)
   end

end 


##

@info("Test Second-order derivatices with Lux")
