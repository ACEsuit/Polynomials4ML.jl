using Polynomials4ML, Test, StaticArrays, LuxCore, Zygote, ForwardDiff
using Polynomials4ML: _generate_input
using Random: default_rng
using ACEbase.Testing: println_slim, print_tf
using LinearAlgebra: dot, I 
using Optimisers: destructure 
import Polynomials4ML as P4ML

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

function _is_equivalent(g1::NamedTuple, g2::NamedTuple)
   for key in keys(g2) 
      if !haskey(g1, key) 
         @error("key $(key) not found in g1")
         return false 
      end
   end
   for key in keys(g1)
      v1 = g1[key] 
      v2 = g2[key]
      if !_is_equivalent(v1, v2)
         @error("key $(key) not equivalent")
         return false 
      end
   end
   return true
end

_is_equivalent(a1::Union{Nothing, @NamedTuple{}}, 
               a2::Union{Nothing, @NamedTuple{}}) = true 

_is_equivalent(a1::AbstractArray, a2::AbstractArray) = (a1 ≈ a2)


##

@info("Testing Lux Layers for basic basis sets")

test_bases = [ chebyshev_basis(10), 
               ChebBasis(8), 
               BernsteinBasis(8), 
               RTrigBasis(10), 
               CTrigBasis(10), 
               MonoBasis(10), 
               legendre_basis(10),
               real_sphericalharmonics(5), 
               real_solidharmonics(5), 
               complex_sphericalharmonics(5), 
               complex_solidharmonics(5), 
               P4ML._rand_gaussian_basis(), 
               P4ML._rand_slater_basis(), 
               P4ML._rand_sto_basis(), 
               ]

##

for basis in test_bases
# basis = test_bases[12]
   @info("Lux layer test for $(typeof(basis).name.name)")
   local B1, B2, x, X, ps, st
   nX = rand(8:16)
   X = [ _generate_input(basis) for _ = 1:nX ]
   B1 = basis(X)
   ps, st = LuxCore.setup(rng, basis)
   B2, _ = basis(X, ps, st)
   println_slim(@test B1 == B2)

   if !( eltype(eltype(B1)) <: Real)
      println("basis is complex; skipping AD test")
      continue 
   end

   # evaluate the basis and get the pullback operator  
   val1, pb1 = Zygote.pullback(LuxCore.apply, basis, X, ps, st)
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
   println_slim(@test ∂1[3] == ∂2[3])

   # look at gradients with respect to the parameters 
   _foo = p -> dot(Δ, LuxCore.apply(basis, X, p, st)[1])
   g1 = Zygote.gradient(_foo, ps)[1] 
   if sizeof(ps) == 0 
      println_slim(@test (isnothing(g1) || isempty(g1)))
   else
      pvec, _restruct = destructure(ps) 
      g2 = _restruct( ForwardDiff.gradient(p -> _foo(_restruct(p)), pvec) )
      println_slim(@test _is_equivalent(g1, g2))
   end

end 

