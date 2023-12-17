
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

function fwddiff1_pfwd(basis::PooledSparseProduct{NB}, BB, ΔBB) where {NB}
   A1 = basis(BB)
   sub_i(t, ti, i) = ntuple(a -> a == i ? ti : t[a], length(t))
   ∂A1_i = [  ForwardDiff.jacobian(B -> basis(sub_i(BB, B, i)), BB[i])
              for i = 1:NB ]
   ∂A1 = sum(∂A1_i[i] * ΔBB[i] for i = 1:NB)            
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

##

A1, ∂A1 = fwddiff_pfwd(basis, BB, ΔBB)
A2, ∂A2 = pfwd(basis, BB, ΔBB)
@show A2 ≈ A1
@show ∂A2 ≈ ∂A1

