
using BenchmarkTools, Test, Polynomials4ML, ChainRulesCore, ForwardDiff
using Polynomials4ML: PooledSparseProduct, evaluate, evaluate!
using ACEbase.Testing: fdtest, println_slim, print_tf
P4ML = Polynomials4ML

##

function _generate_basis(; order=3, len = 50)
   NN = [ rand(10:30) for _ = 1:order ]
   spec = sort([ ntuple(t -> rand(1:NN[t]), order) for _ = 1:len])
   return PooledSparseProduct(spec)
end

function _rand_input1(basis::PooledSparseProduct{ORDER}; 
                      nX = rand(7:12)) where {ORDER} 
   NN = [ maximum(b[i] for b in basis.spec) for i = 1:ORDER ]
   BB = ntuple(i -> randn(nX, NN[i]), ORDER)
   ΔBB = ntuple(i -> randn(nX, NN[i]), ORDER)
   return BB, ΔBB
end

function fwddiff1_pfwd(basis::PooledSparseProduct{3}, BB, ΔBB)
   A1 = basis(BB)
   ∂A1_1 = ForwardDiff.jacobian(B1 -> basis((B1, BB[2], BB[3])), BB[1])
   ∂A1_2 = ForwardDiff.jacobian(B2 -> basis((BB[1], B2, BB[3])), BB[2])
   ∂A1_3 = ForwardDiff.jacobian(B3 -> basis((BB[1], BB[2], B3)), BB[3])
   ∂A1 = ∂A1_1 * ΔBB[1] + ∂A1_2 * ΔBB[2] + ∂A1_3 * ΔBB[3]
   return A1, ∂A1   
end

function fwddiff_pfwd(basis::PooledSparseProduct{NB}, BB, ΔBB) where {NB}
   nX = size(BB[1], 1)
   Aj_∂Aj = [ fwddiff1_pfwd(basis, 
                         ntuple(t ->  BB[t][j,:], NB), 
                         ntuple(t -> ΔBB[t][j,:], NB), ) 
               for j = 1:nX ] 
   Aj = [ x[1] for x in Aj_∂Aj ] 
   ∂Aj = [ x[2] for x in Aj_∂Aj ] 
   A = sum(Aj) 
   ∂A = reduce(hcat, ∂Aj)               
   return A, ∂A
end
##

order = 3
basis = _generate_basis(; order=order)
BB, ΔBB = _rand_input1(basis)

## 


function pfwd(basis::PooledSparseProduct{NB}, BB, ΔBB) where {NB}
   @assert length(size(BB[1])) == 2
   @assert length(size(ΔBB[1])) == 2
   nX = size(ΔBB[1], 1)
   TA = promote_type(eltype.(BB)...)
   A = zeros(TA, length(basis))
   T∂A = promote_type(TA, eltype.(ΔBB)...)
   ∂A = zeros(T∂A, length(basis), nX)
   for (i, ϕ) in enumerate(basis.spec)
      for j = 1:nX 
         bb = ntuple(t -> BB[t][j, ϕ[t]], NB)
         Δbb = ntuple(t -> ΔBB[t][j, ϕ[t]], NB)
         ∏bb, ∇∏bb = Polynomials4ML._static_prod_ed(bb)
         A[i] += prod(bb)
         @inbounds for t = 1:NB
            ∂A[i, j] += ∇∏bb[t] * Δbb[t]
         end
      end
   end
   return A, ∂A
end

@time A1, ∂A1 = fwddiff_pfwd(basis, BB, ΔBB)
@time A2, ∂A2 = pfwd(basis, BB, ΔBB)
A2 ≈ A1
∂A2 ≈ ∂A1

