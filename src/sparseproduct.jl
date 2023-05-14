struct SparseProduct{NB}
   spec::Vector{NTuple{NB, Int}}
   # ---- temporaries & caches 
end

function SparseProduct()
   return SparseProduct(bases, NTuple{NB, Int}[])
end

# each column defines a basis element
function SparseProduct(spec::Matrix{<: Integer})
   @assert all(spec .> 0)
   spect = [ Tuple(spec[:, i]...) for i = 1:size(spec, 2) ]
   return SparseProduct(spect)
end
 
Base.length(basis::SparseProduct) = length(basis.spec)


# ----------------------- evaluation interfaces 


function evaluate(basis::SparseProduct, BB::Tuple{Vararg{AbstractVector}}) 
   VT = mapreduce(eltype, promote_type, BB)
   A = zeros(VT, length(basis))
   evaluate!(A, basis, BB::Tuple)
   return A 
end

function evaluate(basis::SparseProduct, BB::Tuple{Vararg{AbstractMatrix}}) 
   VT = mapreduce(eltype, promote_type, BB)
   nX = size(BB[1], 1)
   A = zeros(VT, nX, length(basis))
   evaluate!(A, basis, BB::Tuple)
   return A 
end

test_evaluate(basis::SparseProduct, BB::Tuple) = 
       [ prod(BB[j][basis.spec[i][j]] for j = 1:length(BB)) 
            for i = 1:length(basis) ]
   

# ----------------------- evaluation kernels 

function evaluate!(A, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractVector}}) where {NB}
   @assert length(BB) == NB
   spec = basis.spec
   for (iA, ϕ) in enumerate(spec)
       @inbounds A[iA] = BB_prod(ϕ, BB)
   end
   return nothing 
end


function evaluate!(A, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractMatrix}}) where {NB}
   nX = size(BB[1], 1)
   @assert all(B->size(B, 1) == nX, BB)
   spec = basis.spec

   @inbounds for (iA, ϕ) in enumerate(spec)
      @simd ivdep for j = 1:nX
         A[j, iA] = BB_prod(ϕ, BB, j)
      end
   end
   return nothing
end

# -------------------- reverse mode gradient

function _rrule_evaluate(basis::SparseProduct{NB}, BB::Tuple) where {NB}
   A = evaluate(basis, BB)
   return A, ∂A -> _pullback_evaluate(∂A, basis, BB)
end


function _pullback_evaluate(∂A, basis::SparseProduct{NB}, BB::Tuple) where {NB}
   TA = promote_type(eltype.(BB)...)
   ∂BB = ntuple(i -> zeros(TA, size(BB[i])...), NB)
   _pullback_evaluate!(∂BB, ∂A, basis, BB)
   return ∂BB
end


function _pullback_evaluate!(∂BB, ∂A, basis::SparseProduct{NB}, BB::Tuple) where {NB}
   nX = size(BB[1], 1)

   @assert all(nX <= size(BB[i], 1) for i = 1:NB)
   @assert all(nX <= size(∂BB[i], 1) for i = 1:NB)
   @assert all(size(∂BB[i], 2) >= size(BB[i], 2) for i = 1:NB)
   @assert size(∂A) == (nX, length(basis))
   @assert length(BB) == NB 
   @assert length(∂BB) == NB
   
   @inbounds for (iA, ϕ) in enumerate(basis.spec) # for each spec
      # ∂A_iA = ∂A[iA]
      @simd ivdep for j = 1:nX 
        b = ntuple(Val(NB)) do i 
           @inbounds BB[i][j, ϕ[i]] 
        end 
        g = _prod_grad(b, Val(NB))
        for i = 1:NB 
           ∂BB[i][j, ϕ[i]] = muladd(∂A[j, iA], g[i], ∂BB[i][j, ϕ[i]])
        end
      end 
   end
   return nothing 
end