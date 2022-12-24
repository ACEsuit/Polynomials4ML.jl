

using Polynomials4ML, BenchmarkTools, StaticArrays

using ObjectPools

using ObjectPools: FlexArrayCache, acquire!, release!

##
N = 30; nX = 256 

basis = OrthPolyBasis1D3T(randn(N), randn(N), randn(N))
X = rand(nX)
P = zeros(nX, N)
pool = FlexArrayCache()

Polynomials4ML.evaluate!(P, basis, X)
Polynomials4ML.evaluate(basis, X)

@info("evaluate 3-term recurrance")
@info("in-place")
@btime Polynomials4ML.evaluate!($P, $basis, $X)
@info("allocating")
@btime Polynomials4ML.evaluate!(zeros(nX, N), $basis, $X)
@info("with pool")
@btime (P = acquire!($pool, (256, 30), Float64); 
        Polynomials4ML.evaluate!(P, $basis, $X); 
        release!(P))

##

basis = CYlmBasis(10)
N = length(basis)
nX = 64
X = randn(SVector{3, Float64}, nX)
Y = zeros(ComplexF64, nX, N)
pool = FlexArrayCache()

Polynomials4ML.evaluate!(Y, basis, X)
Polynomials4ML.evaluate(basis, X)

@info("evaluate 3-term recurrance")
@info("in-place")
@btime Polynomials4ML.evaluate!($Y, $basis, $X)
@info("allocating")
@btime Polynomials4ML.evaluate!(zeros(ComplexF64, nX, N), $basis, $X)
@info("with pool")
@btime (Y = Polynomials4ML.evaluate($basis, $X); release!(Y))
