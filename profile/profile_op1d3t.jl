
using OrthPolys4ML
using OrthPolys4ML: evaluate!, evaluate_ed!, evaluate_ed2!
using BenchmarkTools


##

function time_standard!(P, basis, X)
   for i = 1:length(X)
      evaluate!( (@view P[:, i]), basis, X[i] )
   end
   return P 
end

time_batched!(P, basis, X) = evaluate!(P, basis, X)



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


##

function time_ed_standard!(P, dP, basis, X)
   for i = 1:length(X)
      evaluate_ed!( (@view P[:, i]), (@view dP[:, i]), basis, X[i] )
   end
   return P, dP  
end

time_ed_batched!(P, dP, basis, X) = evaluate_ed!(P, dP, basis, X)



##

dP_standard = zeros(N, nX)
dP_batched = zeros(nX, N)

time_ed_standard!(P_standard, dP_standard, basis, X)
time_ed_batched!(P_batched, dP_batched, basis, X)

@info("evaluate_ed!")
@info("Naive Implementation")
@btime time_ed_standard!($P_standard, $dP_standard, $basis, $X)
@info("Batched Implementation")
@btime time_ed_batched!($P_batched, $dP_batched, $basis, $X)



##

function time_ed2_standard!(P, dP, ddP, basis, X)
   for i = 1:length(X)
      evaluate_ed2!( (@view P[:, i]), (@view dP[:, i]), (@view ddP[:, i]), basis, X[i] )
   end
   return P, dP  
end

time_ed2_batched!(P, dP, ddP, basis, X) = evaluate_ed2!(P, dP, ddP, basis, X)



##

ddP_standard = zeros(N, nX)
ddP_batched = zeros(nX, N)

time_ed2_standard!(P_standard, dP_standard, ddP_standard, basis, X)
time_ed2_batched!(P_batched, dP_batched, ddP_batched, basis, X)

@info("evaluate_ed2!")
@info("Naive Implementation")
@btime time_ed2_standard!($P_standard, $dP_standard, $ddP_standard, $basis, $X)
@info("Batched Implementation")
@btime time_ed2_batched!($P_batched, $dP_batched, $ddP_batched, $basis, $X)

