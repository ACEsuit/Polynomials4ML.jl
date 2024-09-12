
struct Pure2B{TA, TAA} <: AbstractP4MLTensor
   abasis::TA 
   aabasis::TAA 
   # --------------
   @reqfields
end

function Pure2B(abasis, aabasis)
   return Pure2B(abasis, aabasis, _make_reqfields()...)
end

Base.length(basis::Pure2B) = length(basis.aabasis)

Base.show(io::IO, basis::Pure2B{TA, TAA}) where {TA, TAA} = 
      print(io, "Pure2B($(TA.name.name), $(TAA.name.name))")

# -------------- evaluation interfaces

_valtype(basis::Pure2B, args...) = _valtype(basis.abasis, args...)

_gradtype(basis::Pure2B, args...) = _gradtype(basis.abasis, args...)

_generate_input(basis::Pure2B; nX = rand(5:15)) = 
   _generate_input(basis.abasis; nX = nX)

function _generate_batch(basis::Pure2B, args...; kwargs...) 
   error("Pure2B is not implemented for batch inputs")
end

function whatalloc(::typeof(evaluate!), basis::Pure2B, 
                   BB::Tuple{Matrix{T1}, Matrix{T2}}) where {T1, T2}
   VT = promote_type(T1, T2) 
   return (VT, length(basis))
end

# ----------------------- evaluation kernels 

# specialized implementation for two inputs only
# the prototype is Rnl, Ylm. The comments in this function are copied 
# from the AA basis implementation. 

function evaluate!(AA, basis::Pure2B, BB::Tuple{TB1, TB2}) where {TB1, TB2}
   aspec = basis.abasis.spec
   nodes = basis.aabasis.nodes
   has0 = basis.aabasis.has0
   num1 = basis.aabasis.num1
   @assert length(AA) >= basis.aabasis.numstore

   B1, B2 = BB
   @assert size(B1, 1) == size(B2, 1) 
   nX = size(B1, 1)

   TAA = eltype(AA)

   if has0
      error("Pure2B does not support has0 == true")
   end 

   PHI = zeros(TAA, nX, length(nodes))

   # Stage-1: copy the 1-particle basis into AA
   # note this entirely ignores the spec / nodes. It is implicit in the 
   # definitions and orderings
   @inbounds for i = 1:num1
      # AA[has0+i] = A[i]
      n1, n2 = aspec[i] 
      ai = zero(TAA) 
      for j = 1:nX 
         PHI[j, i] = ϕ_ij = B1[j, n1] * B2[j, n2]
         ai += ϕ_ij
      end
      AA[i] = ai 
   end

   # Stage-2: go through the dag and store the intermediate results we need
   @inbounds for i = (num1+has0+1):length(nodes)
      n1, n2 = nodes[i]
      # AA[i] = AA[n1] * AA[n2]
      aai = zero(TAA)
      for j = 1:nX 
         PHI[j, i] = ϕ_ij = PHI[j, n1] * PHI[j, n2]
         aai += ϕ_ij
      end 
      AA[i] = aai
   end

   return AA
end
