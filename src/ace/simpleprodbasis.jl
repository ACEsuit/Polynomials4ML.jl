

"""
Naive implementation of the product basis, intended only for testing
"""
struct SimpleProdBasis  <: AbstractP4MLTensor
   orders::Vector{Int}
   spec::Matrix{Int} 
end 

Base.length(basis::SimpleProdBasis) = size(basis.spec, 1)


function SimpleProdBasis(specv::AbstractVector{<: AbstractVector}) 
   @assert issorted(length.(specv))
   @assert all(issorted, specv)
   orders = [length(s) for s in specv]
   maxord = maximum(orders)
   specm = zeros(Int, length(specv), maxord)
   for i = 1:length(specv)
      specm[i, 1:orders[i]] = specv[i]
   end
   return SimpleProdBasis(orders, specm)
end 


# ----------------------- evaluation code 
#   we only provide the evaluate functionality itself to test the DAG 
#   gradients can just be checked by finite differences

_valtype(basis::SimpleProdBasis, ::Type{T}) where {T} = T

function whatalloc(::typeof(evaluate!), 
                   basis::SimpleProdBasis, A::AbstractVector{T}) where {T}
   VT = _valtype(basis, T)
   return (VT, length(basis))
end




function evaluate!(AA, basis::SimpleProdBasis, A::AbstractVector)
   for i = 1:length(basis) 
      AA[i] = prod( A[basis.spec[i, a]]  for a = 1:basis.orders[i]; 
                   init = one(eltype(AA)) )
   end
   return AA 
end

