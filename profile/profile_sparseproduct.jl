using Polynomials4ML, BenchmarkTools

##

N1 = 10 
N2 = 20 
N3 = 50 
spec = sort([ (rand(1:N1), rand(1:N2), rand(1:N3)) for i = 1:100 ])

basis = Polynomials4ML.SparseProduct(spec)

@info("Test inplace evaluation")
nX = 64 
bBB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )

function serial_evaluate!(A, basis, bBB) 
   fill!(A, 0.0)
   nX = size(bBB[1], 1)
   for i = 1:nX
      evaluate!(A[i, :], basis, ntuple(j -> @view(bBB[j][i, :]), length(bBB)))
   end
   return nothing
end 


A1 = zeros(Float64, nX, length(basis))
A2 = zeros(Float64, nX, length(basis))

@assert A1 ≈ A2
@info("timing serial and batched")
@btime serial_evaluate!($A1, $basis, $bBB)
@btime Polynomials4ML.evaluate!($A2, $basis, $bBB)

##

# @profview let A2=A2, basis=basis, bBB=bBB
#    for _=1:3_000_000
#       Polynomials4ML.evaluate!(A2, basis, bBB)
#    end
# end

# ##

# @profview let A2=A2, basis=basis, bBB=bBB
#    for _=1:3_000_000
#       Polynomials4ML.evaluate(basis, bBB)
#    end
# end



## 

∂A = randn(size(A1))
val, pb = Polynomials4ML._rrule_evaluate(basis, bBB)
∂BB = pb(∂A)

@info("timing pullback")
@btime begin 
   val, pb = Polynomials4ML._rrule_evaluate($basis, $bBB)
   ∂BB = pb($∂A)
end

##

@info("timing in-place pullback")
display( @benchmark Polynomials4ML._pullback_evaluate!($∂BB, $∂A, $basis, $bBB) )

##