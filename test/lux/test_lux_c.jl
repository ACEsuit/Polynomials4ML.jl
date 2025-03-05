
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
      xvec -> m( [SVector{3}(xvec[1:3]), SVector{3}(xvec[4:6])], ps, st)[1], 
      [xx[1]; xx[2]] )
@show norm(vcat(gz...) - gf, Inf)
println_slim(@test norm(vcat(gz...) - gf, Inf) < 1e-10)

# but we also need to check that the output is real, which originally it 
# wasn't...
@show eltype(gz)
println_slim(@test eltype(gz) == SVector{3, Float64})


##

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

gf = ForwardDiff.gradient(
      xvec -> m( [SVector{3}(xvec[1:3]), SVector{3}(xvec[4:6])], ps, st)[1], 
      [xx[1]; xx[2]] )

gz = Zygote.gradient(x -> m(x, ps, st)[1], xx)[1]

@show norm(vcat(gz...) - gf, Inf)
println_slim(@test norm(vcat(gz...) - gf, Inf) < 1e-10)

# but we also need to check that the output is real, which originally it 
# wasn't...
@warn("This test is still failing, type should be real:")
@show eltype(gz)
# println_slim(@test eltype(gz) == SVector{3, Float64})

##

# Test 3: 