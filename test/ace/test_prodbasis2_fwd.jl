
using Test, BenchmarkTools, Polynomials4ML
using Polynomials4ML: SimpleProdBasis, release!, SparseSymmProd
using Polynomials4ML.Testing: println_slim, print_tf, generate_SO2_spec
using Random

using ACEbase.Testing: fdtest, dirfdtest
using Lux, ForwardDiff

P4ML = Polynomials4ML
##

M = 5
spec = generate_SO2_spec(5, 2*M)

A = randn(Float64, 2*M+1)
nX = 10
ΔA = randn(length(A), nX)

basis = SparseSymmProd(spec)
AA = basis(A)


## 



function pfwd(basis::SparseSymmProd, A, ΔA)
   @assert !basis.hasconst
   nAA = length(basis)
   nX = size(ΔA, 2)
   AA = zeros(eltype(A), nAA)
   ∂AA = zeros(eltype(ΔA), nAA, nX)

   function _pfwd_in(::Val{N}, rg_N, spec_N::Vector{NTuple{N, Int}}) where {N}
      for (i, bb) in zip(rg_N, spec_N)
         aa = ntuple(t -> A[bb[t]], N)
         ∏aa, ∇∏aa = Polynomials4ML._static_prod_ed(aa)
         AA[i] = ∏aa
         for t = 1:N, j = 1:nX
            ∂AA[i, j] += ∇∏aa[t] * ΔA[bb[t], j]
         end
      end
   end 

   Base.Cartesian.@nexprs 5 N -> _pfwd_in(Val{N}(), basis.ranges[N], basis.specs[N])

   return AA, ∂AA
end

@time ∂AA1 = ForwardDiff.jacobian(basis, A) * ΔA
@time AA2, ∂AA2 = pfwd(basis, A, ΔA)

norm(∂AA1 - ∂AA2, Inf)

