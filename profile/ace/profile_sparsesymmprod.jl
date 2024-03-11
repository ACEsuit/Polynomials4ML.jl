
using Test, BenchmarkTools, Polynomials4ML
using Polynomials4ML: SimpleProdBasis, release!, SparseSymmProd
using Polynomials4ML.Testing: generate_SO2_spec
using Random
using ChainRulesCore: rrule 
using StrideArrays
# using Zygote

P4ML = Polynomials4ML

## 

function _gen_a_spec(D)
   a_spec = [ (n=n, l=l, m=m) for n = 1:D+1 for l = 0:D for m = -l:l
                               if n-1+l <= D ]
   _deg = b -> b.n - 1 + b.l 
   a_spec_D = _deg.(a_spec)
   p = sortperm(a_spec_D)::Vector{Int}
   a_spec_D = a_spec_D[p]
   a_spec = a_spec[p]
   return a_spec, a_spec_D 
end 

function generate_O3_spec(valN::Val{N}, D) where {N}
   a_spec, a_spec_D = _gen_a_spec(D)

   tup2b = let a_spec=a_spec 
      vv -> tuple([a_spec[vi] for vi in vv[vv .> 0]]...)
   end
   admissible = let D = D 
         bb -> (length(bb) == 0) || (sum( b.n-1+b.l for b in bb ) <= D)
   end
   filter = bb -> (length(bb) > 0) && iseven(sum(b.l for b in bb)) && iszero(sum(b.m for b in bb))
   maxvv = [ length(a_spec) for _=1:N ]

   aa_spec = P4ML.Utils.gensparse(; NU=N, maxvv=maxvv, tup2b=tup2b, admissible=admissible, filter=filter, ordered=true)
   
   return a_spec, aa_spec 
end

##

D = 10; N = 3 
a_spec, _ = _gen_a_spec(D)


##

D = 14; N = 4
a_spec, aa_spec = generate_O3_spec(Val(N), D)
aa_spec = [ vv[vv .> 0] for vv in aa_spec ]
@show length(a_spec) 
@show length(aa_spec)

##

function eval_loop!(AA, basis, A)
   N_b = size(A, 2)
   for i = 1:N_b 
      evaluate!((@view AA[:, i]), basis, (@view A[:, i]))
   end
end

function pb_loop!(∂A, ΔAA, basis, AA)
   N_b = size(AA, 2)
   for i = 1:N_b 
      P4ML.pullback_arg!((@view ∂A[:, i]), (@view ΔAA[:, i]), basis, (@view AA[:, i]))
   end
end


basis = SparseSymmProdDAG(aa_spec)
n_AA = length(basis(randn(length(a_spec))))

println("----------------------------")
println("General Info")
println("   O(3)-invariants")
println("   #A  = $(length(a_spec))")
println("   #AA = $(length(aa_spec))")

for N_b in [8, 16, 32]
   println("----------------------------")
   println("#batch = $N_b")

   A    = PtrArray(randn(length(a_spec), N_b))
   AA   = PtrArray(zeros(n_AA, N_b))
   A_b  = PtrArray(collect(A'))
   AA_b = PtrArray(zeros(N_b, n_AA))

   eval_loop!(AA, basis, A)
   evaluate!(AA_b, basis, A_b)
   # @show AA ≈ AA_b'

   println("Forward Pass")
   print("   in loop: "); @btime eval_loop!($AA, $basis, $A)
   print("   batched: "); @btime evaluate!($AA_b, $basis, $A_b)

   ΔAA   = PtrArray(randn(size(AA)))
   ΔAA_b = PtrArray(collect(ΔAA'))
   ∂A    = PtrArray(zeros(size(A)))
   ∂A_b  = PtrArray(zeros(size(A_b)))

   pb_loop!(∂A, ΔAA, basis, AA)
   P4ML.pullback_arg!(∂A_b, ΔAA_b, basis, AA_b)
   # @show ∂A ≈ ∂A_b'

   println("Backward Pass")
   print("   in loop: "); @btime pb_loop!($∂A, $ΔAA, $basis, $AA)
   print("   batched: "); @btime P4ML.pullback_arg!($∂A_b, $ΔAA_b, $basis, $AA_b)
end
println("----------------------------")
