
using Polynomials4ML, Test, StaticArrays, Lux 
using Polynomials4ML: lux
using Random: default_rng
using LinearAlgebra: dot, norm
using ACEbase.Testing: println_slim
using Zygote, ForwardDiff
rng = default_rng()
P4ML = Polynomials4ML
complex_sphericalharmonics

##
# auxiliary functions
function _2svec3(xx::AbstractVector{<: Number}) 
   N = length(xx) ÷ 3 
   return [ SVector{3}(xx[(i-1)*3+1:i*3]) for i = 1:N ]
end

function _2vec(xx::AbstractVector{<: SVector})
   return vcat([ xx[i] for i = 1:length(xx) ]...)
end


## --------------------------------------------------------------------- 
#
# Test 1: generate a Lux chain embedding complex Ylms, then 
#         inner product with a real vector, then take real part.
# 
# This test fails when naive complex multiplication is used instead of 
# treating the product of complex numbers as an inner product.

L = 3 
bY = complex_sphericalharmonics(L)
θ = randn(2, length(bY))

# NOTE: These tests fail if the chain is constructed slightly differently:
# m = Chain( Y = lux(bY), 
#            dot = WrappedFunction(y -> real(dot(θ, y))) )

m = Chain( Y = lux(bY), 
           dot = WrappedFunction(y -> dot(θ, real.(y))) )
ps, st = Lux.setup(rng, m)

xx = [ randn(SVector{3, Float64}), randn(SVector{3, Float64}) ]
y, _ = m(xx, ps, st)

gz = Zygote.gradient(x -> m(x, ps, st)[1], xx)[1]

gf = ForwardDiff.gradient(
               xvec -> m( _2svec3(xvec), ps, st)[1], _2vec(xx) )
@show norm(_2vec(gz) - gf, Inf)
println_slim(@test norm(vcat(gz...) - gf, Inf) < 1e-10)

# but we also need to check that the output is real, which originally it 
# wasn't...
@show eltype(gz)
println_slim(@test eltype(gz) == SVector{3, Float64})


## --------------------------------------------------------------------- 
#
# Test 2: generate a Lux chain embedding complex Ylms, then 
#         inner product with a COMPLEX vector, then take real part
#         to still ensure real output hence real gradients.

L = 3 
bY = complex_sphericalharmonics(L)
θ = randn(ComplexF64, 2, length(bY))

# NOTE: These tests fail if the chain is constructed slightly differently:
# m = Chain( Y = lux(bY), 
#            dot = WrappedFunction(y -> real(dot(θ, y))) )

m = Chain( Y = lux(bY), 
           dot = WrappedFunction(y -> sum(θ .* y) ), 
           re = WrappedFunction(y -> real(y) ) )
ps, st = Lux.setup(rng, m)

xx = [ randn(SVector{3, Float64}), randn(SVector{3, Float64}) ]
y, _ = m(xx, ps, st)

gf = ForwardDiff.gradient( xvec -> m( _2svec3(xvec), ps, st)[1], _2vec(xx) )

gz = Zygote.gradient(x -> m(x, ps, st)[1], xx)[1]

@show norm(_2vec(gz) - gf, Inf)
println_slim(@test norm(vcat(gz...) - gf, Inf) < 1e-10)

# but we also need to check that the output is real, which originally it 
# wasn't...
@warn("This test is still failing, type should be real:")
@show eltype(gz)
# println_slim(@test eltype(gz) == SVector{3, Float64})

## --------------------------------------------------------------------- 
#
# Test 3: 
#

function _generate_basis(; order=3, len = 30)
   NN = [ rand(10:20) for _ = 1:order ]
   spec = sort([ ntuple(t -> rand(1:NN[t]), order) for _ = 1:len])
   return PooledSparseProduct(spec)
end

bA = _generate_basis(; order = 2)
bY = complex_sphericalharmonics(4)
bR = legendre_basis(21)
θ = randn(length(bA))

_embed = Parallel(nothing; 
                  Y = lux(bY), 
                  R = Chain(nrm = WrappedFunction(X -> norm.(X)), 
                             Rn = lux(bR)) )
m = Chain( embed = _embed, 
           A = lux(bA),
           dot = WrappedFunction(a -> dot(θ, real.(a))) )

ps, st = Lux.setup(rng, m)

xx = [ rand(SVector{3, Float64}) .- 0.5 for _ = 1:5 ]
m(xx, ps, st)

gf = ForwardDiff.gradient(
      xvec -> m( _2svec3(xvec), ps, st)[1], 
      _2vec(xx) )

gz = Zygote.gradient(x -> m(x, ps, st)[1], xx)[1]

@show norm(_2vec(gz) - gf, Inf)
println_slim(@test norm(_2vec(gz) - gf, Inf) < 1e-10)

# check the output is real 
@show eltype(gz)
println_slim(@test eltype(gz) == SVector{3, Float64})

## ---------------------------------------------------------------------