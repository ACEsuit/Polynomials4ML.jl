
using Polynomials4ML, BenchmarkTools
using Polynomials4ML.Testing: time_standard!, time_batched!, 
                     time_ed_standard!, time_ed_batched!, 
                     time_ed2_standard!, time_ed2_batched!

##
N = 30; nX = 256 

basis = OrthPolyBasis1D3T(randn(N), randn(N), randn(N))
X = rand(nX)
P_standard = zeros(N, nX)
P_batched = zeros(nX, N)

time_standard!(P_standard, basis, X)
time_batched!(P_batched, basis, X)

@info("evaluate!")
@info("Naive Implementation")
@btime time_standard!($P_standard, $basis, $X)
@info("Batched Implementation")
@btime time_batched!($P_batched, $basis, $X)



# ##

# dP_standard = zeros(N, nX)
# dP_batched = zeros(nX, N)

# time_ed_standard!(P_standard, dP_standard, basis, X)
# time_ed_batched!(P_batched, dP_batched, basis, X)

# @info("evaluate_ed!")
# @info("Naive Implementation")
# @btime time_ed_standard!($P_standard, $dP_standard, $basis, $X)
# @info("Batched Implementation")
# @btime time_ed_batched!($P_batched, $dP_batched, $basis, $X)


# ##

# ddP_standard = zeros(N, nX)
# ddP_batched = zeros(nX, N)

# time_ed2_standard!(P_standard, dP_standard, ddP_standard, basis, X)
# time_ed2_batched!(P_batched, dP_batched, ddP_batched, basis, X)

# @info("evaluate_ed2!")
# @info("Naive Implementation")
# @btime time_ed2_standard!($P_standard, $dP_standard, $ddP_standard, $basis, $X)
# @info("Batched Implementation")
# @btime time_ed2_batched!($P_batched, $dP_batched, $ddP_batched, $basis, $X)

