using Polynomials4ML
using BenchmarkTools


SUITE = BenchmarkGroup()


## Test polynomials

SUITE["Polynomials"] = BenchmarkGroup()

N = 100
Np = 10
r = 2*rand(N) .- 1
tmp = zeros(N,N)
tmp_d = similar(tmp)
tmp_d2 = similar(tmp)

# Chebyshev
ch_basis = ChebBasis(Np)

SUITE["Polynomials"]["Chebyshev"] = BenchmarkGroup()
SUITE["Polynomials"]["Chebyshev"]["evaluation"] = @benchmarkable evaluate!($tmp, $ch_basis, $r)
SUITE["Polynomials"]["Chebyshev"]["derivative"] = @benchmarkable evaluate_ed!($tmp, $tmp_d, $ch_basis, $r)
SUITE["Polynomials"]["Chebyshev"]["2nd derivative"] = @benchmarkable evaluate_ed2!($tmp, $tmp_d, $tmp_d2, $ch_basis, $r)