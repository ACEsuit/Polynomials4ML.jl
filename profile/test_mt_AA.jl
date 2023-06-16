
using BenchmarkTools, Test, Polynomials4ML
using Polynomials4ML: PooledSparseProduct, evaluate, evaluate!, BB_prod,    
                      SimpleProdBasis, release!, SparseSymmProd
using StrideArrays, LoopVectorization, Tullio, Polyester
using Main.Threads
using Polynomials4ML.Testing: generate_SO2_spec

P4ML = Polynomials4ML


##

M = 15
spec = generate_SO2_spec(5, 2*M)
A = randn(Float64, 2*M+1)

##

