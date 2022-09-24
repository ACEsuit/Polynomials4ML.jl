
using Polynomials4ML, BenchmarkTools
using Polynomials4ML.Testing: time_standard!, time_batched!
##

L = 8; nX = 1024
basis = CYlmBasis(L)
X = [Polynomials4ML.rand_sphere() for _=1:nX]
P_batched = evaluate(basis, X)
P_standard = copy(P_batched')

time_standard!(P_standard, basis, X)
time_batched!(P_batched, basis, X)

##

@info("evaluate!")
@info("Naive Implementation")
@btime time_standard!($P_standard, $basis, $X)
@info("Batched Implementation")
@btime time_batched!($P_batched, $basis, $X)

## 

@profview let basis = basis, X = X, P_batched = P_batched 
   for n = 1:30_000 
      time_batched!(P_batched, basis, X)
   end
end

##

alp = basis.alp
S = Polynomials4ML.cart2spher.(X)

P_batched = evaluate(alp, S)
P_standard = copy(P_batched')

time_standard!(P_standard, alp, S)
time_batched!(P_batched, alp, S) 

##
@info("evaluate!")
@info("Naive Implementation")
@btime time_standard!($P_standard, $alp, $S)
@info("Batched Implementation")
@btime time_batched!($P_batched, $alp, $S)

##

@profview let alp = alp, S = S, P_batched = P_batched 
   for n = 1:100_000 
      time_batched!(P_batched, alp, S)
   end
end

##

@btime Polynomials4ML.evaluate!($P_batched, $alp, $S)
