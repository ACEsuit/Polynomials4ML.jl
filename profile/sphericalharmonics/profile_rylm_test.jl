## quick performance test 

using BenchmarkTools
using Polynomials4ML: release!

maxL = 10 
nX = 32 
rSH = RYlmBasis(maxL)
cSH = CYlmBasis(maxL)
X = [ rand_sphere() for i = 1:nX ]

@info("quick performance test")
@info("Real, single input")
@btime ( Y = evaluate($rSH, $(X[1])); release!(Y); )
@info("Real, $nX inputs")
@btime ( Y = evaluate($rSH, $X); release!(Y); )
@info("Complex, $nX inputs")
@btime ( Y = evaluate($cSH, $X); release!(Y); )

@info("Real, grad, single input")
@btime begin Y, dY = evaluate_ed($rSH, $(X[1])); release!(Y); release!(dY); end 
@info("Real, grad, $nX inputs")
@btime begin Y, dY = evaluate_ed($rSH, $X); release!(Y); release!(dY); end
@info("Complex, $nX inputs")
@btime begin Y, dY = evaluate_ed($cSH, $X); release!(Y); release!(dY); end

@info("laplacian, batched")
@btime begin ΔY = $(P4.laplacian)($rSH, $X); release!(ΔY); end 

@info("eval_grad_laplace")
@btime begin Y, dY, ΔY = $(P4.eval_grad_laplace)($rSH, $X); release!(Y); release!(dY); release!(ΔY); end 

