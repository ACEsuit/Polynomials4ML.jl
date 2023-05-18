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
   
function evaluate_ed(basis::SparseProduct, BB::Tuple{Vararg{AbstractVector}}, ∂BB::Tuple{Vararg{AbstractVector}}) 
   VT = mapreduce(eltype, promote_type, ∂BB)
   A, dA = zeros(VT, length(basis)), zeros(VT, length(basis))
   evaluate_ed!(A, dA, basis, BB::Tuple, ∂BB::Tuple)
   return A, dA
end

function evaluate_ed(basis::SparseProduct, BB::Tuple{Vararg{AbstractMatrix}}, ∂BB::Tuple{Vararg{AbstractMatrix}}) 
   VT = mapreduce(eltype, promote_type, ∂BB)
   nX = size(∂BB[1], 1)
   A, dA = zeros(VT, nX, length(basis)), zeros(VT, nX, length(basis))
   evaluate_ed!(A, dA, basis, BB::Tuple, ∂BB::Tuple)
   return A, dA
end

function evaluate_ed2(basis::SparseProduct, BB::Tuple{Vararg{AbstractVector}}, ∂BB::Tuple{Vararg{AbstractVector}}, ∂∂BB::Tuple{Vararg{AbstractVector}}) 
   VT = mapreduce(eltype, promote_type, ∂∂BB)
   A = zeros(VT, length(basis))
   dA = zeros(VT, length(basis))
   ddA = zeros(VT, length(basis))
   evaluate_ed2!(A, dA, ddA, basis, BB::Tuple, ∂BB::Tuple, ∂∂BB::Tuple)
   return A, dA, ddA
end

function evaluate_ed2(basis::SparseProduct, BB::Tuple{Vararg{AbstractMatrix}}, ∂BB::Tuple{Vararg{AbstractMatrix}}, ∂∂BB::Tuple{Vararg{AbstractMatrix}}) 
   VT = mapreduce(eltype, promote_type, ∂∂BB)
   nX = size(∂∂BB[1], 1)
   A = zeros(VT, nX, length(basis))
   dA = zeros(VT, nX, length(basis))
   ddA = zeros(VT, nX, length(basis))
   evaluate_ed2!(A, dA, ddA, basis, BB::Tuple, ∂BB::Tuple, ∂∂BB::Tuple)
   return A, dA, ddA
end
# ----------------------- evaluation kernels 

function evaluate!(A, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractVector}}) where {NB}
   @assert length(BB) == NB
   spec = basis.spec
   for (iA, ϕ) in enumerate(spec)
       @inbounds A[iA] = BB_prod(ϕ, BB)
   end
   return A 
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
   return A
end

# Not sure whether we can everything below
# faster by eval and diff at the same time from prod_grad

function evaluate_ed!(A, dA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractVector}}, ∂BB::Tuple{Vararg{AbstractVector}}) where {NB}
   @assert length(BB) == NB
   @assert length(∂BB) == NB
   spec = basis.spec
   # evaluate!(A, basis, BB)
   for (iA, ϕ) in enumerate(spec)
      b = ntuple(Val(NB)) do i 
         @inbounds BB[i][ϕ[i]] 
      end 
      g = _prod_grad_ed(b, Val(NB))
      A[iA] = g[1]
      for i = 1:NB
         dA[iA] = muladd(∂BB[i][ϕ[i]], g[i + 1], dA[iA])
      end
   end 
   return A, dA 
end

function evaluate_ed!(A, dA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractMatrix}}, ∂BB::Tuple{Vararg{AbstractMatrix}}) where {NB}
   nX = size(BB[1], 1)
   @assert all(B->size(B, 1) == nX, BB)
   @assert all(∂B->size(∂B, 1) == nX, ∂BB)
   spec = basis.spec
   # evaluate!(A, basis, BB)
   @inbounds for (iA, ϕ) in enumerate(spec)
      @simd ivdep for j = 1:nX 
        b = ntuple(Val(NB)) do i 
           @inbounds BB[i][j, ϕ[i]] 
        end 
        g = _prod_grad_ed(b, Val(NB))
        A[j, iA] = g[1] 
        for i = 1:NB
           dA[j, iA] = muladd(∂BB[i][j, ϕ[i]], g[i + 1], dA[j, iA])
        end
      end 
   end
   return A, dA
end

function evaluate_ed2!(A, dA, ddA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractVector}}, ∂BB::Tuple{Vararg{AbstractVector}}, ∂∂BB::Tuple{Vararg{AbstractVector}}) where {NB}
   @assert length(BB) == NB
   @assert length(∂BB) == NB
   @assert length(∂∂BB) == NB
   spec = basis.spec

   evaluate_ed!(A, dA, basis, BB, ∂BB)

   for (iA, ϕ) in enumerate(spec)
      b = ntuple(Val(NB)) do i 
         @inbounds BB[i][ϕ[i]] 
      end 
      dg = _prod_grad(b, Val(NB))
      for i = 1:NB 
         ddA[iA] = muladd(∂∂BB[i][ϕ[i]], dg[i], ddA[iA])
      end
      for m = 1:NB-1
         for n = m+1:NB
            @inbounds ddA[iA] += 2 * BB2_prod(ϕ, BB, ∂BB, m, n)
         end
      end
   end 
   return A, dA, ddA 
end

function evaluate_ed2!(A, dA, ddA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractMatrix}}, ∂BB::Tuple{Vararg{AbstractMatrix}}, ∂∂BB::Tuple{Vararg{AbstractMatrix}}) where {NB}
   nX = size(BB[1], 1)
   @assert all(B->size(B, 1) == nX, BB)
   @assert all(∂B->size(∂B, 1) == nX, ∂BB)
   @assert all(∂∂B->size(∂∂B, 1) == nX, ∂∂BB)
   spec = basis.spec

   evaluate_ed!(A, dA, basis, BB, ∂BB)
   
   @inbounds for (iA, ϕ) in enumerate(spec)
      @simd ivdep for j = 1:nX
         b = ntuple(Val(NB)) do i 
            @inbounds BB[i][j, ϕ[i]] 
         end 
         g = _prod_grad(b, Val(NB))
         for i = 1:NB 
            ddA[j, iA] = muladd(∂∂BB[i][j, ϕ[i]], g[i], ddA[j, iA])
         end
         for m = 1:NB-1
            for n = m+1:NB
               @inbounds ddA[j, iA] += 2 * BB2_prod(ϕ, BB, ∂BB, j, m, n)
            end
         end
      end
   end
   return A, dA, ddA
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

test_evaluate(basis::SparseProduct, BB::Tuple) = 
       [ prod(BB[j][basis.spec[i][j]] for j = 1:length(BB)) 
            for i = 1:length(basis) ]


function test_evaluate_ed(basis::SparseProduct, BB::Tuple, ∂BB::Tuple) 
   A = zeros(length(basis))
   eval = [ prod(BB[j][basis.spec[i][j]] for j = 1:length(BB)) 
            for i = 1:length(basis) ]
   for i = 1:length(basis)
      for j = 1:length(BB)
         A[i] += eval[i]/BB[j][basis.spec[i][j]] * ∂BB[j][basis.spec[i][j]]
      end
   end
   return A
end 

function test_evaluate_ed2(basis::SparseProduct, BB::Tuple, ∂BB::Tuple, ∂∂BB::Tuple) 
   A = zeros(length(basis))
   eval = [ prod(BB[j][basis.spec[i][j]] for j = 1:length(BB)) 
            for i = 1:length(basis) ]
   for i = 1:length(basis)
      for j = 1:length(BB)
         A[i] += eval[i]/BB[j][basis.spec[i][j]] * ∂∂BB[j][basis.spec[i][j]]
      end
      for j = 1:length(BB)-1
         for z = j+1:length(BB)
            A[i] += 2 * eval[i]/(BB[j][basis.spec[i][j]]*BB[z][basis.spec[i][z]]) * ∂BB[j][basis.spec[i][j]] * ∂BB[z][basis.spec[i][z]]
         end
      end
   end
   return A
end 