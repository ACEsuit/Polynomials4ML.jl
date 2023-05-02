
using Polynomials4ML, StaticArrays, BenchmarkTools, LinearAlgebra, LoopVectorization
using Polynomials4ML: index_y

L = 5
bYlm = RYlmBasis(L)
bRlm = RRlmBasis(L)

rr = randn( SVector{3, Float64}, 100)
_Y = evaluate(bYlm, rr)
Y = collect(_Y)
rl = zeros(length(rr), bYlm.alp.L+1)
drl = zeros(length(rr), bYlm.alp.L+1)
dY = collect(evaluate_ed(bYlm, rr)[2])

function solids!(Y, bYlm, rr, rl)

   @inbounds begin 
   # compute r^l
   Nr = length(rr)
   @simd ivdep for i = 1:Nr
      rl[i, 1] = 1.0  # l = 0
      rl[i, 2] = norm(rr[i])  # l = 1
   end
   for l = 2:bYlm.alp.L
      @simd ivdep for i = 1:Nr
         rl[i, l+1] = rl[i, l] * rl[i, 2]
      end
   end

   # evaluate the spherical harmonics 
   evaluate!(Y, bYlm, rr)

   # rescale them 
   for l = 0:bYlm.alp.L, m = -l:l
      iY = index_y(l, m)
      @simd ivdep for i = 1:Nr
         Y[i, iY] *= rl[i, l+1]
      end
   end

   end

   return nothing 
end

function solids_ed!(Y, dY, bYlm, rr, rl, drl)
   Nr = length(rr)

   @inbounds begin 
   # compute r^l
   @simd ivdep for i = 1:Nr
      rl[i, 1] = 1.0  # l = 0
      drl[i, 1] = 0.0 
      rl[i, 2] = norm(rr[i])  # l = 1
      drl[i, 2] = 1.0 
   end
   for l = 2:bYlm.alp.L
      @simd ivdep for i = 1:Nr
         rl[i, l+1] = rl[i, l] * rl[i, 2]
         drl[i, l+1] = l * rl[i, l]
      end
   end

   # evaluate the spherical harmonics 
   evaluate_ed!(Y, dY, bYlm, rr)

   # rescale them 
   for l = 0:bYlm.alp.L, m = -l:l
      iY = index_y(l, m)
      @simd ivdep for i = 1:Nr
         Y[i, iY] *= rl[i, l+1]
         dY[i, iY] = rl[i, l+1] * dY[i, iY] + (drl[i, l+1] * Y[i, iY] / rl[i, 2]) * rr[i] 
      end
   end

   end

   return nothing 
end

##

@info("Evaluate spherical harmonics")
@btime evaluate!($Y, $bYlm, $rr)
@info("Evaluate solid harmonics wrapper")
@btime solids!($Y, $bYlm, $rr, $rl)
@info("Evaluate Dexuan's harmonics")
@btime evaluate!($Y, $bRlm, $rr)

##

@info("Gradient spherical harmonics")
@btime evaluate_ed!($Y, $dY, $bYlm, $rr)
@info("Gradient solid harmonics wrapper")
@btime solids_ed!($Y, $dY, $bYlm, $rr, $rl, $drl)
@info("Gradient Dexuan's harmonics")
@btime evaluate_ed!($Y, $dY, $bRlm, $rr)


##

# finding the bottleneck... 

@profview let Y = Y, dY = dY, bRlm = bRlm, rr = rr 
   for _ = 1:100_000 
      evaluate_ed!(Y, dY, bRlm, rr)
   end
end