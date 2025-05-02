# RETIRE 

export SparseProduct

"""
`SparseProduct` : a model layer to build tensor products
"""
struct SparseProduct{NB} <: AbstractP4MLTensor
   spec::Vector{NTuple{NB, Int}}
   # ---- temporaries & caches
   @reqfields()
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

SparseProduct(spec) = SparseProduct(spec, _make_reqfields()...)

_valtype(basis::SparseProduct, BB::Tuple) = 
      mapreduce(eltype, promote_type, BB)

function _generate_input_1(basis::SparseProduct{NB}) where {NB} 
   NN = [ maximum(b[i] for b in basis.spec) for i = 1:NB ]
   BB = ntuple(i -> randn(NN[i]), NB)
   return BB 
end 

function _generate_input(basis::SparseProduct{NB}; nX = rand(5:15)) where {NB} 
   NN = [ maximum(b[i] for b in basis.spec) for i = 1:NB ]
   BB = ntuple(i -> randn(nX, NN[i]), NB)
   return BB 
end 

# ----------------------- overiding alloc functions
# specifically for SparseProduct/PooledSparseProduct

_out_size(basis::SparseProduct, BB::TupVec) = (length(basis), )
_out_size(basis::SparseProduct, BB::TupMat) = (size(BB[1],1), length(basis))

_out_size(basis::SparseProduct, BB::Tuple{AbstractVector, AbstractVector}) = (length(basis), )
_out_size(basis::SparseProduct, BB::Tuple{AbstractMatrix, AbstractMatrix}) = (size(BB[1],1), length(basis))

# ----------------------- evaluation kernels 

function whatalloc(::typeof(evaluate!), basis::SparseProduct{NB}, BB::TupVecMat) where {NB}
   VT = _valtype(basis, BB)
   return (VT, _out_size(basis, BB)...)
end

function evaluate!(A, basis::SparseProduct{NB}, 
                   BB::TupVec) where {NB}
   @assert length(BB) == NB
   spec = basis.spec
   for (iA, ϕ) in enumerate(spec)
      b = ntuple(t->BB[t][ϕ[t]], NB)
      @inbounds A[iA] = @fastmath prod(b)
   end
   return A 
end

function evaluate!(A, basis::SparseProduct{NB}, 
                   BB::TupMat) where {NB}
   nX = size(BB[1], 1)
   @assert all(B->size(B, 1) == nX, BB)
   spec = basis.spec

   @inbounds for (iA, ϕ) in enumerate(spec)
      @simd ivdep for j = 1:nX
         b = ntuple(t->BB[t][j, ϕ[t]], NB)
         A[j, iA] = @fastmath prod(b)
      end
   end
   return A
end

# special-casing NB = 1 for correctness 
function evaluate!(A, basis::SparseProduct{1}, 
                   BB::Tuple{<: AbstractMatrix},
                   nX = size(BB[1], 1))
   @assert size(BB[1], 1) >= nX
   BB1 = BB[1] 
   spec = basis.spec
   fill!(A, zero(eltype(A)))
   @inbounds for (iA, ϕ) in enumerate(spec)
      ϕ1 = ϕ[1]
      @simd ivdep for j = 1:nX
         A[j, iA] = BB1[j, ϕ1]
      end
   end
   return A
end

# -------------------- reverse mode gradient

function whatalloc(::typeof(pullback!), 
                   ∂A, basis::SparseProduct{NB}, BB::TupMat) where  {NB}
   TA = promote_type(eltype.(BB)..., eltype(∂A))
   return ntuple(i -> (TA, size(BB[i])...), NB)                   
end

# adapt to WithAlloc, should be sufficiently for now up to NB = 4
# but should be later replaced by generated code
pullback!(∂B1::AbstractMatrix, ∂A, basis::SparseProduct{1}, BB::TupMat) = 
         pullback!((∂B1,), ∂A, basis, BB)

pullback!(∂B1, ∂B2, ∂A, basis::SparseProduct{2}, BB::TupMat) = 
         pullback!((∂B1, ∂B2,), ∂A, basis, BB)

pullback!(∂B1, ∂B2, ∂B3, ∂A, basis::SparseProduct{3}, BB::TupMat) = 
         pullback!((∂B1, ∂B2, ∂B3,), ∂A, basis, BB)

pullback!(∂B1, ∂B2, ∂B3, ∂B4, ∂A, basis::SparseProduct{4}, BB::TupMat) = 
         pullback!((∂B1, ∂B2, ∂B3, ∂B4,), ∂A, basis, BB)

# NB = 1 for correctness 
function pullback!(∂BB::Tuple, ∂A, basis::SparseProduct{1}, BB::TupMat)
   nX = size(BB[1], 1)
   NB = 1
   @assert size(∂A) == (nX, length(basis))
   @assert length(BB) == length(∂BB) == NB 
   @assert all(nX <= size(BB[i], 1) for i = 1:NB)
   @assert all(nX <= size(∂BB[i], 1) for i = 1:NB)
   @assert all(size(∂BB[i], 2) >= size(BB[i], 2) for i = 1:NB)
   BB1 = BB[1]
   ∂BB1 = ∂BB[1]

   fill!(∂BB1, zero(eltype(∂BB1)))
   
   @inbounds for (iA, ϕ) in enumerate(basis.spec)
      ϕ1 = ϕ[1]
      @simd ivdep for j = 1:nX 
         ∂BB1[j, ϕ1] += ∂A[j, iA]
      end 
   end
   return ∂BB 
end

function pullback!(∂BB, ∂A, basis::SparseProduct{NB}, BB::Tuple) where {NB}
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
        a, g = _static_prod_ed(b)
        for i = 1:NB 
           ∂BB[i][j, ϕ[i]] = muladd(∂A[j, iA], g[i], ∂BB[i][j, ϕ[i]])
        end
      end 
   end
   return ∂BB 
end




# ----------------------- evaluation interfaces 

function pushforward(basis::SparseProduct, 
                              BB::Tuple{Vararg{AbstractVector}}, 
                              ∂BB::Tuple{Vararg{AbstractVector}}) 
   VT = mapreduce(eltype, promote_type, BB)
   A = zeros(VT, length(basis))
   # ∂BB: Vector of SVector{3, Float64}
   # dA: Matrix 3 * length(basis)
   dA = zeros(VT, length(∂BB[1][1]), length(basis)) 
   pushforward!(A, dA, basis, BB::Tuple, ∂BB::Tuple)
   return A, dA
end

function pushforward(basis::SparseProduct, 
                              BB::Tuple{Vararg{AbstractMatrix}}, 
                              ∂BB::Tuple{Vararg{AbstractMatrix}}) 
   VT = mapreduce(eltype, promote_type, BB)
   nX = size(∂BB[1], 1)
   # BB: Matrix Nel * length(basis)
   # ∂BB: Matrix of SVector{3, Float64}: Nel * length(basis)
   A = zeros(VT, nX, length(basis))
   dA = [zeros(VT, length(∂BB[1][1])) for i = 1:nX, j = 1:length(basis)]
   pushforward!(A, dA, basis, BB::Tuple, ∂BB::Tuple)
   return A, dA
end

#=
function _frule_frule_evaluate(basis::SparseProduct, BB::Tuple{Vararg{AbstractVector}}, ∂BB::Tuple{Vararg{AbstractVector}}, ∂∂BB::Tuple{Vararg{AbstractVector}}) 
   VT = mapreduce(eltype, promote_type, BB)
   A = zeros(VT, length(basis))
   # ∂BB: Vector of SVector{3, Float64}
   # dA: Matrix 3 * length(basis)
   dA = zeros(VT, length(∂BB[1][1]), length(basis)) 
   ddA = zeros(VT, length(∂BB[1][1]), length(basis)) 
   _frule_frule_evaluate!(A, dA, ddA, basis, BB::Tuple, ∂BB::Tuple, ∂∂BB::Tuple)
   return A, dA, ddA
end

function _frule_frule_evaluate(basis::SparseProduct, BB::Tuple{Vararg{AbstractMatrix}}, ∂BB::Tuple{Vararg{AbstractMatrix}}, ∂∂BB::Tuple{Vararg{AbstractMatrix}}) 
   VT = mapreduce(eltype, promote_type, BB)
   nX = size(∂BB[1], 1)
   # BB: Matrix Nel * length(basis)
   # ∂BB: Matrix of SVector{3, Float64}: Nel * length(basis)
   A = zeros(VT, nX, length(basis))
   dA = [zeros(VT, length(∂BB[1][1])) for i = 1:nX, j = 1:length(basis)]
   ddA = [zeros(VT, length(∂BB[1][1])) for i = 1:nX, j = 1:length(basis)]
   _frule_frule_evaluate!(A, dA, ddA, basis, BB::Tuple, ∂BB::Tuple, ∂∂BB::Tuple)
   return A, dA, ddA
end
=#

function pushforward!(A, dA, basis::SparseProduct{NB}, 
                               BB::Tuple{Vararg{AbstractVector}}, 
                               ∂BB::Tuple{Vararg{AbstractVector}}) where {NB}
   @assert length(BB) == NB
   @assert length(∂BB) == NB
   spec = basis.spec
   # evaluate!(A, basis, BB)
   for (iA, ϕ) in enumerate(spec)
      b = ntuple(Val(NB)) do i 
         @inbounds BB[i][ϕ[i]] 
      end 
      a, g = _static_prod_ed(b)
      A[iA] = a
      for i = 1:NB
         for j = 1:length(∂BB[1][1])
            dA[j, iA] = muladd(∂BB[i][ϕ[i]][j], g[i], dA[iA])
         end
      end
   end 
   return A, dA 
end

function pushforward!(A, dA, basis::SparseProduct{NB}, 
                               BB::Tuple{Vararg{AbstractMatrix}}, 
                               ∂BB::Tuple{Vararg{AbstractMatrix}}) where {NB}
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
        a, g = _static_prod_ed(b)
        A[j, iA] = a 
        for i = 1:NB
            for k = 1:length(∂BB[1][1])
               dA[j, iA][k] = muladd(∂BB[i][j, ϕ[i]][k], g[i], dA[j, iA])
            end
        end
      end 
   end
   return A, dA
end

#=
function _frule_frule_evaluate!(A, dA, ddA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractVector}}, ∂BB::Tuple{Vararg{AbstractVector}}, ∂∂BB::Tuple{Vararg{AbstractVector}}) where {NB}
   @assert length(BB) == NB
   @assert length(∂BB) == NB
   @assert length(∂∂BB) == NB
   spec = basis.spec

   for (iA, ϕ) in enumerate(spec)
      b = ntuple(Val(NB)) do i 
         @inbounds BB[i][ϕ[i]] 
      end 
      g = _prod_ed2(b, Val(NB))
      A[iA] = g[1]
      for i = 1:NB 
         for j = 1:length(∂BB[1][1])
            dA[iA, j] = muladd(∂BB[i][ϕ[i]][j], g[i + 1], dA[iA])
            ddA[iA, j] = muladd(∂∂BB[i][ϕ[i]][j], g[i + 1], ddA[iA])
         end
      end
      t = 1
      for m = 1:NB-1
         for n = m+1:NB
            for j = 1:length(∂BB[1][1])
               ddA[iA, j] = muladd(2 * ∂BB[m][ϕ[m]][j] * ∂BB[n][ϕ[n]][j], g[t + 1 + NB], ddA[iA])
            end
            t += 1
         end
      end
   end 
   return A, dA, ddA 
end

function _frule_frule_evaluate!(A, dA, ddA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractMatrix}}, ∂BB::Tuple{Vararg{AbstractMatrix}}, ∂∂BB::Tuple{Vararg{AbstractMatrix}}) where {NB}
   nX = size(BB[1], 1)
   @assert all(B->size(B, 1) == nX, BB)
   @assert all(∂B->size(∂B, 1) == nX, ∂BB)
   @assert all(∂∂B->size(∂∂B, 1) == nX, ∂∂BB)
   spec = basis.spec
   
   @inbounds for (iA, ϕ) in enumerate(spec)
      @simd ivdep for j = 1:nX
         b = ntuple(Val(NB)) do i 
            @inbounds BB[i][j, ϕ[i]] 
         end 
         g = _prod_ed2(b, Val(NB))
         A[j, iA] = g[1]
         for i = 1:NB 
            for k = 1:length(∂BB[1][1])
               dA[j, iA][k] = muladd(∂BB[i][j, ϕ[i]][k], g[i + 1], dA[j, iA])
               ddA[j, iA][k] = muladd(∂∂BB[i][j, ϕ[i]][k], g[i + 1], ddA[j, iA])
            end
         end
         t = 1
         for m = 1:NB-1
            for n = m+1:NB
               ddA[j, iA][k] = muladd(2 * ∂BB[m][j, ϕ[m]][k] * ∂BB[n][j, ϕ[n]][k], g[t + 1 + NB], ddA[j, iA])
            end
            t += 1
         end
      end
   end
   return A, dA, ddA
end

=#

# --------------------- connect with ChainRules 
# can this be generalized again? 

import ChainRulesCore: rrule, NoTangent

function rrule(::typeof(evaluate), basis::SparseProduct{NB}, BB::TupMat) where {NB}
   A = evaluate(basis, BB)

   function pb(Δ)
      ∂BB = pullback(Δ, basis, BB)
      return NoTangent(), NoTangent(), ∂BB
   end 

   return A, pb 
end
