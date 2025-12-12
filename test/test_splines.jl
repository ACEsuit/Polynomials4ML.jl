


using Polynomials4ML, Test, LuxCore, Random 
using Polynomials4ML: _generate_input
using Polynomials4ML.Testing: println_slim, print_tf, test_all 
using LinearAlgebra: I, norm, dot 
using QuadGK
import Polynomials4ML as P4ML 


##

# N = rand(5:15)
N = 10 
# basis = OrthPolyBasis1D3T(randn(N), randn(N), randn(N))
basis = chebyshev_basis(N)
spec = Polynomials4ML.natural_indices(basis)

##

using StaticArrays
import Interpolations as INT

Nnod = 30 
xx = range(-1.0, 1.0; length=Nnod)
PP = [ SVector{length(basis)}(basis(x)) for x in xx ]

spl = INT.cubic_spline_interpolation(xx, PP) 

st_spl1 = P4ML.splinify(basis, -1.0, 1.0, Nnod; bspline=true)
st_spl2 = P4ML.splinify(basis, -1.0, 1.0, Nnod; bspline=false)

##

Random.seed!(1234)

for x in xx 
   P1 = basis(x)
   P2 = spl(x)
   P3 = st_spl1(x)
   print_tf(@test P1 ≈ P2 ≈ P3) 
end
println() 


for x in xx[1:end-1] 
   y = x + rand() * 2/(Nnod-1) 
   P1 = basis(y)
   P2 = spl(y)
   P3 = st_spl1(y)
   print_tf(@test P2 ≈ P3)
   scalerr = norm( (P1 - P2) * (1-y^2)^2 ./ (1:N).^3 , Inf)
   print_tf(@test scalerr < 1e-5)
end
println() 


##

P4ML.Testing.test_all(st_spl1; ka = true)


##

# import ForwardDiff as FD

# st_spl1 = SPL.splinify(basis, -1.0, 1.0, 30)
# st_spl2 = SPL.splinify(spl, -1.0, 0.99999, 30)

# x = 1 - rand()/(Nnod-1)
# P1 = basis(x)
# P2 = spl(x)
# P3 = st_spl1(x)
# P4 = st_spl2(x)

# @show norm(P1 - P2, Inf)
# @show norm(P1 - P3, Inf)
# @show norm(P2 - P3, Inf)
# @show norm(P2 - P4, Inf)
# @show norm(P3 - P4, Inf)


##

# TODO: add benchmaks 

# using BenchmarkTools

# function _benchrun(basis, Ntest = 1000)
#    s = 0.0 
#    for ntest = 1:Ntest 
#       x = rand() * 2 - 1
#       P1 = basis(x)
#       s += P1[2] 
#    end
#    return s 
# end 

# function _benchrun_psst(basis, ps, st, Ntest = 1000)
#    s = 0.0 
#    for ntest = 1:Ntest 
#       x = rand() * 2 - 1
#       P1, st = basis(x, ps, st)
#       s += P1[2] 
#    end
#    return s 
# end 


# @btime _benchrun($basis)
# @btime _benchrun($spl)
# @btime _benchrun($st_spl1)

# ps, st = LuxCore.setup(MersenneTwister(1234), st_spl1)
# @btime _benchrun_psst($st_spl1, $ps, $st)

# st_spl4 = P4ML.splinify(basis, -1.0, 1.0, Nnod; bspline=true)
# ps4, st4 = LuxCore.setup(MersenneTwister(1234), st_spl4)
# @btime _benchrun_psst($st_spl4, $ps4, $st4)

