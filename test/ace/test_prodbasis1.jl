
using Test, BenchmarkTools, Polynomials4ML
using Polynomials4ML: SimpleProdBasis, release!, SparseSymmProd
using Polynomials4ML.Testing: println_slim, print_tf, generate_SO2_spec
using LoopVectorization, Polyester, StrideArrays, Tullio
using ACEbase.Testing: fdtest, dirfdtest

P4ML = Polynomials4ML
##


function generate_spec(nA, order) 
   spec = Vector{Int}[] 
   # order - 1
   append!(spec, [ [n,] for n = 1:nA ])
   # order - 2
   append!(spec, [ [n, m] for n = 1:nA for m = n:nA ])
   # order - 3
   append!(spec, [ [n, m, l] for n = 1:nA for m = n:nA for l = m:nA ])
   #
   return return filter(bb -> sum(bb) <= nA, spec)
end

struct AASpec 
   spec1::Vector{Int} 
   spec2::Vector{Tuple{Int, Int}}
   spec3::Vector{Tuple{Int, Int, Int}}
end 

struct AASpecTullio
   spec1::Vector{Int} 
   spec2::Matrix{Int}
   spec3::Matrix{Int}
end 

function AASpecTullio(spec::AASpec)
   spec2 = reshape(reinterpret(Int, spec.spec2), 2, :)
   spec3 = reshape(reinterpret(Int, spec.spec3), 3, :)
   return AASpecTullio(spec.spec1, Matrix(spec2), Matrix(spec3))
end

function eval_simd!(AA, spec::AASpec, A)
   n1 = length(spec.spec1)
   n2 = length(spec.spec2)
   n3 = length(spec.spec3)
   nX = size(A, 1)
   @inbounds begin 
   for i = 1:n1
      @simd ivdep for j = 1:nX
         AA[j, i] = A[j, i]
      end
   end
   @inbounds for i = n1+1:n2
      b = spec.spec2[i]
      @simd ivdep for j = 1:nX
         AA[j, i] = A[j, b[1]] * A[j, b[2]]
      end
   end
   @inbounds for i = n2+1:n3
      b = spec.spec3[i]
      @simd ivdep for j = 1:nX
         AA[j, i] = A[j, b[1]] * A[j, b[2]] * A[j, b[3]]
      end
   end
   end 
   return nothing 
end

function eval_poly!(AA, spec::AASpec, A)
   n1 = length(spec.spec1)
   n2 = length(spec.spec2)
   n3 = length(spec.spec3)
   nX = size(A, 1)
   @inbounds begin 
   @batch for i = 1:n1
      @simd ivdep for j = 1:nX
         AA[j, i] = A[j, i]
      end
   end
   @batch for i = n1+1:n2
      b = spec.spec2[i]
      @simd ivdep for j = 1:nX
         AA[j, i] = A[j, b[1]] * A[j, b[2]]
      end
   end
   @batch for i = n2+1:n3
      b = spec.spec3[i]
      @simd ivdep for j = 1:nX
         AA[j, i] = A[j, b[1]] * A[j, b[2]] * A[j, b[3]]
      end
   end
   end 
   return nothing 
end

function eval_tullio!(AA, spec::AASpec, A)
   n1 = length(spec.spec1)
   n2 = length(spec.spec2)
   n3 = length(spec.spec3)
   n12 = n1+n2
   nX = size(A, 1)
   AA1 = view(AA, :, 1:n1)
   AA2 = view(AA, :, n1+1:n2)
   AA3 = view(AA, :, n2+1:n3)
   b2 = spec.spec2
   b3 = spec.spec3

   @inbounds begin 
      @tullio AA1[j, i] = A[j, i]
      @tullio AA2[j, i] = A[j, b2[1, n1+i]] * A[j, b2[2, n1+i]]
      @tullio AA3[j, i] = A[j, b3[1, n12 + i]] * A[j, b3[2, n12 + i]] * A[j, b3[3, n12 + i]]
   end 
   return nothing 
end


##

nA = 150
order = 3 
spec = generate_spec(nA, order)

spec1 = [ n[1] for n in spec[ length.(spec) .== 1 ] ]
spec2 = map(bb -> tuple(bb...), spec[ length.(spec) .== 2 ])
spec3 = map(bb -> tuple(bb...), spec[ length.(spec) .== 3 ])
aaspec = AASpec(spec1, spec2, spec3)
tlspec = AASpecTullio(aaspec)

basis = SparseSymmProd(spec)

##

nX = 32
A = randn(Float64, nX, nA)
AA0 = basis(A)
AA1 = copy(AA0)

eval_simd!(AA1, aaspec, A)
eval_poly!(AA1, aaspec, A)
# eval_tullio!(AA1, aaspec, A)

##

@info("Performance of order-3 AA basis, 32 inputs")
@info("P4ML (serial, simd across inputs)")
@btime evaluate!($AA0, $basis, $A) 
@info("Slightly optimized (serial, simd across inputs)")
@btime eval_simd!($AA1, $aaspec, $A)
@info("Polyester Threads (simd across inputs, @batch across outputs)")
@btime eval_poly!($AA1, $aaspec, $A)

@info("Compare this against $nX x single eval:")
println("318 * nX / nthreads = ", 318 * nX / Threads.nthreads() / 1000, "ms")
A_ = A[1,:] 
AA0_ = AA0[1,:]
@btime evaluate!($AA0_, $basis, $A_)
