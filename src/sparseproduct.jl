using ChainRulesCore
using ChainRulesCore: NoTangent

"""
`SparseProduct` : a model layer to build tensor products
"""
struct SparseProduct{NB} <: AbstractPoly4MLBasis
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

_valtype(basis::SparseProduct{T1}, TX::NTuple{NB, AbstractVecOrMat{T2}}) where {T1, T2, NB} = T2

# ----------------------- evaluation interfaces 
function _frule_evaluate(basis::SparseProduct, BB::Tuple{Vararg{AbstractVector}}, ∂BB::Tuple{Vararg{AbstractVector}}) 
   VT = mapreduce(eltype, promote_type, BB)
   A = zeros(VT, length(basis))
   # ∂BB: Vector of SVector{3, Float64}
   # dA: Matrix 3 * length(basis)
   dA = zeros(VT, length(∂BB[1][1]), length(basis)) 
   _frule_evaluate!(A, dA, basis, BB::Tuple, ∂BB::Tuple)
   return A, dA
end

function _frule_evaluate(basis::SparseProduct, BB::Tuple{Vararg{AbstractMatrix}}, ∂BB::Tuple{Vararg{AbstractMatrix}}) 
   VT = mapreduce(eltype, promote_type, BB)
   nX = size(∂BB[1], 1)
   # BB: Matrix Nel * length(basis)
   # ∂BB: Matrix of SVector{3, Float64}: Nel * length(basis)
   A = zeros(VT, nX, length(basis))
   dA = [zeros(VT, length(∂BB[1][1])) for i = 1:nX, j = 1:length(basis)]
   _frule_evaluate!(A, dA, basis, BB::Tuple, ∂BB::Tuple)
   return A, dA
end

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

# ----------------------- overiding alloc functions
# specifically for SparseProduct/PooledSparseProduct
_outsym(x::TupVec) = :out
_outsym(X::TupMat) = :outb

# TODO: generalize it
#_outsym(x::Tuple{AbstractVector, AbstractVector}) = :out
#_outsym(X::Tuple{AbstractMatrix, AbstractMatrix}) = :outb

_out_size(basis::SparseProduct, BB::TupVec) = (length(basis), )
_out_size(basis::SparseProduct, BB::TupMat) = (size(BB[1],1), length(basis))

_out_size(basis::SparseProduct, BB::Tuple{AbstractVector, AbstractVector}) = (length(basis), )
_out_size(basis::SparseProduct, BB::Tuple{AbstractMatrix, AbstractMatrix}) = (size(BB[1],1), length(basis))

function _alloc_d(basis::SparseProduct, BBs::NTuple{NB, AbstractVecOrMat{T}}) where {NB, T}
      BBs_size = [size(bb) for bb in BBs]
      return [Tuple([acquire!(basis.pool, _outsym(BBs), (BBsize), _valtype(basis, BBs)) for BBsize in BBs_size]) for _ = 1:length(basis)]
end

function _alloc_dd(basis::SparseProduct, BBs::NTuple{NB, AbstractVecOrMat{T}}) where {NB, T}
      BBs_size = [size(bb) for bb in BBs]
      return [Tuple([acquire!(basis.pool, _outsym(BBs), (BBsize), _valtype(basis, BBs)) for BBsize in BBs_size]) for _ = 1:length(basis)]
end

_alloc_ed(basis::SparseProduct, x::NTuple{NB, AbstractVecOrMat{T}}) where {NB, T} = _alloc(basis, x), _alloc_d(basis, x)
_alloc_ed2(basis::SparseProduct, x::NTuple{NB, AbstractVecOrMat{T}}) where {NB, T} = _alloc(basis, x), _alloc_d(basis, x), _alloc_dd(basis, x)



# ----------------------- evaluation kernels 

function evaluate!(A, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractVector}}) where {NB}
   @assert length(BB) == NB
   spec = basis.spec
   for (iA, ϕ) in enumerate(spec)
      b = ntuple(t->BB[t][ϕ[t]], NB)
      @inbounds A[iA] = @fastmath prod(b)
   end
   return A 
end

function evaluate!(A, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractMatrix}}) where {NB}
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


function evaluate_ed!(A, dA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractVector}}) where {NB}
   @assert length(BB) == NB
   spec = basis.spec
   # evaluate!(A, basis, BB)
   for (iA, ϕ) in enumerate(spec)
      b = ntuple(Val(NB)) do i 
         @inbounds BB[i][ϕ[i]] 
      end 
      a, g = _static_prod_ed(b)
      A[iA] = a
      fill!.(dA[iA], 0.0)
      for i = 1:NB
         dA[iA][i][ϕ[i]] += g[i]
      end
   end 
   return A, dA 
end

function evaluate_ed!(A, dA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractMatrix}}) where {NB}
   nX = size(BB[1], 1)
   @assert all(B->size(B, 1) == nX, BB)
   spec = basis.spec
   # evaluate!(A, basis, BB)
   @inbounds for (iA, ϕ) in enumerate(spec)
      fill!.(dA[iA], 0.0)
      @simd ivdep for j = 1:nX 
        b = ntuple(Val(NB)) do i 
           @inbounds BB[i][j, ϕ[i]] 
        end 
        a, g = _static_prod_ed(b)
        A[j, iA] = a 
        for i = 1:NB
           dA[iA][i][j, ϕ[i]] += g[i]
        end
      end 
   end
   return A, dA
end

function evaluate_ed2!(A, dA, ddA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractVector}}) where {NB}
   @assert length(BB) == NB
   spec = basis.spec

   for (iA, ϕ) in enumerate(spec)
      b = ntuple(Val(NB)) do i 
         @inbounds BB[i][ϕ[i]] 
      end 
      a, g = _static_prod_ed(b)
      A[iA] = a
      fill!.(dA[iA], 0.0)
      fill!.(ddA[iA], 0.0)
      for i = 1:NB 
         dA[iA][i][ϕ[i]] += g[i]
      end
   end 
   return A, dA, ddA 
end


function evaluate_ed2!(A, dA, ddA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractMatrix}}) where {NB}
   nX = size(BB[1], 1)
   @assert all(B->size(B, 1) == nX, BB)
   spec = basis.spec
   # evaluate!(A, basis, BB)
   @inbounds for (iA, ϕ) in enumerate(spec)
      fill!.(dA[iA], 0.0)
      fill!.(ddA[iA], 0.0)
      @simd ivdep for j = 1:nX 
        b = ntuple(Val(NB)) do i 
           @inbounds BB[i][j, ϕ[i]] 
        end 
        a, g = _static_prod_ed(b)
        A[j, iA] = a
        for i = 1:NB
           dA[iA][i][j, ϕ[i]] += g[i]
        end
      end 
   end
   return A, dA
end

function _frule_evaluate!(A, dA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractVector}}, ∂BB::Tuple{Vararg{AbstractVector}}) where {NB}
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

function _frule_evaluate!(A, dA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractMatrix}}, ∂BB::Tuple{Vararg{AbstractMatrix}}) where {NB}
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
# -------------------- reverse mode gradient

function ChainRulesCore.rrule(::typeof(evaluate), basis::SparseProduct{NB}, BB::Tuple) where {NB}
   A = evaluate(basis, BB)
   function pb(∂A)
      return NoTangent(), NoTangent(), _pullback_evaluate(∂A, basis, BB)
   end
   return A, pb
end


# function _rrule_evaluate(basis::SparseProduct{NB}, BB::Tuple) where {NB}
#    A = evaluate(basis, BB)
#    return A, ∂A -> _pullback_evaluate(∂A, basis, BB)
# end


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
        a, g = _static_prod_ed(b)
        for i = 1:NB 
           ∂BB[i][j, ϕ[i]] = muladd(∂A[j, iA], g[i], ∂BB[i][j, ϕ[i]])
        end
      end 
   end
   return nothing 
end


