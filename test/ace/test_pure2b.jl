using Test, Polynomials4ML, ChainRulesCore
using Polynomials4ML: SparseSymmProdDAG, 
                      evaluate, evaluate_d, evaluate_dd
using Polynomials4ML.Testing: println_slim, print_tf, generate_SO2_spec
using Random

using ACEbase.Testing: fdtest, dirfdtest

P4ML = Polynomials4ML

maxn = 8; maxl = 4; 
aspec = [ (n, l) for n = 1:maxn for l = 1:maxl ]
abasis = PooledSparseProduct(aspec)
lena = length(aspec)
aaspec = [ [ [ k, ] for k = 1:lena ]; 
           [ [k1, k2]  for k1 = 1:lena for k2 = k1:lena ]; 
           [ [k1, k2, k3] for k1 = 1:lena for k2 = k1:lena for k3 = k2:lena ] ]
aaspec = filter( bb -> sum(bb) <= 20, aaspec)
aabasis = SparseSymmProdDAG(aaspec)
basis = P4ML.Pure2B(abasis, aabasis)

I1 = findall(length.(aaspec) .== 1)
I2 = findall(length.(aaspec) .== 2)
I3 = findall(length.(aaspec) .== 3)

##

nX = 12
R = randn(nX, maxn) 
Y = randn(nX, maxl) 
A = basis.abasis((R, Y))
AA = basis.aabasis(A)
ϕ = evaluate(basis, (R, Y))
AA2 = AA - ϕ

##

@info("confirm permutation invariance")
p = shuffle(1:nX)
Rp = R[p, :]
Yp = Y[p, :]
Ap = basis.abasis((Rp, Yp))
AAp = basis.aabasis(Ap)
ϕp = evaluate(basis, (Rp, Yp))
AA2p = AAp - ϕp
println_slim(@test AA2 ≈ AA2p) 

##

# to show that there is no 2B we can write it explicitly as sum over clusters 
# the actual 2B (1-corr) terms should just be zero now 
@info("Confirm purified 1-corr")
println_slim(@test all(abs.(AA2[I1]) .<= 1e-14))

@info("Confirm purified 2-corr")
# for 3B / 2-Corr we sum over all clusters 
AA_cl2 = zeros(size(AA2[I2]))
K1 = [ bb[1] for bb in aaspec[I2] ]
K2 = [ bb[2] for bb in aaspec[I2] ]
for j1 = 1:nX, j2 = 1:nX 
   if j1 != j2 
      ϕ1 = [ R[j1, n] * Y[j1, l] for (n, l) in aspec ]
      ϕ2 = [ R[j2, n] * Y[j2, l] for (n, l) in aspec ]
      AA_cl2 += ϕ1[K1] .* ϕ2[K2]
   end
end
println_slim(@test AA_cl2 ≈ AA2[I2])

@info("Confirm purified 3-corr")
# same for 4B / 3-corr
AA_cl3 = zeros(size(AA2[I3]))
K1 = [ bb[1] for bb in aaspec[I3] ]
K2 = [ bb[2] for bb in aaspec[I3] ]
K3 = [ bb[3] for bb in aaspec[I3] ]
for j1 = 1:nX, j2 = 1:nX, j3 = 1:nX
   if !(j1 == j2 == j3)
      ϕ1 = [ R[j1, n] * Y[j1, l] for (n, l) in aspec ]
      ϕ2 = [ R[j2, n] * Y[j2, l] for (n, l) in aspec ]
      ϕ3 = [ R[j3, n] * Y[j3, l] for (n, l) in aspec ]
      AA_cl3 += ϕ1[K1] .* ϕ2[K2] .* ϕ3[K3]
   end
end
println_slim(@test AA_cl3 ≈ AA2[I3])

