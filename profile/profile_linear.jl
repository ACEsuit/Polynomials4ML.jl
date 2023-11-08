using Polynomials4ML, BenchmarkTools, LuxCore, Random
using ObjectPools, Zygote

in_d, out_d = 1000, 100 # feature dimensions
N = 10 # batch size

@info("feature_first = true")
l = Polynomials4ML.LinearLayer(in_d, out_d; feature_first = false)
ps, st = LuxCore.setup(MersenneTwister(1234), l)

X = rand(N, in_d)
release!(X)
@btime $l($X, $ps, $st)

# @profview let l = l, ps = ps, st = st, X = X
#    for _ = 1:100_000
#       out = l(X, ps, st)[1]
#       release!(out)
#    end
# end

@info("feature_first = true")
l = Polynomials4ML.LinearLayer(in_d, out_d; feature_first = true)
ps, st = LuxCore.setup(MersenneTwister(1234), l)

X = rand(in_d, N)
release!(X)
@btime $l($X, $ps, $st)

# @profview let l = l, ps = ps, st = st, X = X
#    for _ = 1:100_000
#       out = l(X, ps, st)[1]
#       release!(out)
#    end
# end

@info("benchmark gradient")
out = l(X, ps, st)
Zygote.pullback(x -> l(x, ps, st)[1], X)







