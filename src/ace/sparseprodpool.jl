using ChainRulesCore
using ChainRulesCore: NoTangent

export PooledSparseProduct 


@doc raw"""
`struct PooledSparseProduct` : 
This implements a fused (tensor) product and pooling operation. Suppose 
we are given $N$ embeddings $\phi^{(i)}_{k_i}$ then the pooled sparse product 
generates feature vectors of the form 
```math 
A_{k_1, \dots, k_N} = \sum_{j} \prod_{t = 1}^N \phi^{(t)}_{k_t}(x_j)
```
where $x_j$ are an list of inputs (multi-set). 

### Constructor 
```julia
PooledSparseProduct(spec)
```
where `spec` is a list of $(k_1, \dots, k_N)$ tuples or vectors, or 
`AbstractMatrix` where each column specifies such a tuple. 
"""
struct PooledSparseProduct{NB} <: AbstractPoly4MLBasis
   spec::Vector{NTuple{NB, Int}}
   # ---- temporaries & caches 
   @reqfields
end

function PooledSparseProduct()
   return PooledSparseProduct(NTuple{NB, Int}[], _make_reqfields()...)
end

function PooledSparseProduct(spect::AbstractVector{<: Tuple})
   return PooledSparseProduct(spect, _make_reqfields()...)
end

# each column defines a basis element
function PooledSparseProduct(spec::AbstractMatrix{<: Integer})
   @assert all(spec .> 0)
   spect = [ Tuple(spec[:, i]...) for i = 1:size(spec, 2) ]
   return PooledSparseProduct(spect)
end

Base.length(basis::PooledSparseProduct) = length(basis.spec)

function Base.show(io::IO, basis::PooledSparseProduct{NB}) where {NB}
   print(io, "PooledSparseProduct{$NB}(...)")
end


# ----------------------- evaluation interfaces 
_valtype(basis::AbstractPoly4MLBasis, BB::Tuple) = 
      mapreduce(eltype, promote_type, BB)

_gradtype(basis::AbstractPoly4MLBasis, BB::Tuple) = 
      mapreduce(eltype, promote_type, BB)

_alloc(basis::PooledSparseProduct, BB::TupVecMat) = 
      acquire!(basis.pool, :A, (length(basis), ), _valtype(basis, BB) )


# _alloc_d(basis::AbstractPoly4MLBasis, BB::TupVecMat) = 
#       acquire!(basis.pool, _outsym(BB), (length(basis), ), _gradtype(basis, BB) )

# _alloc_dd(basis::AbstractPoly4MLBasis, BB::TupVecMat) = 
#       acquire!(basis.pool, _outsym(BB), (length(basis), ), _gradtype(basis, BB) )

# _alloc_ed(basis::AbstractPoly4MLBasis, BB::TupVecMat) = 
#       _alloc(basis, BB), _alloc_d(basis, BB)

# _alloc_ed2(basis::AbstractPoly4MLBasis, BB::TupVecMat) = 
#       _alloc(basis, BB), _alloc_d(basis, BB), _alloc_dd(basis, BB)



# ----------------------- evaluation kernels 

import Base.Cartesian: @nexprs

# # Stolen from KernelAbstractions
# import Adapt 
# struct ConstAdaptor end
# import Base.Experimental: @aliasscope
# Adapt.adapt_storage(::ConstAdaptor, a::Array) = Base.Experimental.Const(a)
# constify(arg) = Adapt.adapt(ConstAdaptor(), arg)


# Valentin Churavy's Version (which we don't really understand)
# function evaluate!(A, basis::PooledSparseProduct{NB}, BB) where {NB}
#    @assert length(BB) == NB
#    # evaluate the 1p product basis functions and add/write into _A
#    BB = constify(BB)
#    spec = constify(basis.spec)
#    @aliasscope begin # No store to A aliases any read from any B
#       for (iA, ϕ) in enumerate(spec)
#          @inbounds A[iA] += BB_prod(ϕ, BB)
#       end
#    end
#    return nothing 
# end

function evaluate!(A, basis::PooledSparseProduct{NB}, BB::TupVec) where {NB}
   @assert length(BB) == NB
   # evaluate the 1p product basis functions and add/write into _A
   spec = basis.spec
   fill!(A, 0)
   for (iA, ϕ) in enumerate(spec)
      b = ntuple(t -> BB[t][ϕ[t]], NB)
      @inbounds A[iA] += @fastmath prod(b) 
   end
   return nothing 
end


# # BB::tuple of matrices 
# function evalpool!(A, basis::PooledSparseProduct{NB}, BB, 
#                    nX = size(BB[1], 1)) where {NB}
#    @assert all(B->size(B, 1) >= nX, BB)
#    BB = constify(BB) # Assumes that no B aliases A
#    spec = constify(basis.spec)

#    @aliasscope begin # No store to A aliases any read from any B
#       @inbounds for (iA, ϕ) in enumerate(spec)
#          a = zero(eltype(A))
#          @simd ivdep for j = 1:nX
#             a += BB_prod(ϕ, BB, j)
#          end
#          A[iA] = a
#       end
#    end
#    return nothing
# end

# BB::tuple of matrices 
function evaluate!(A, basis::PooledSparseProduct{NB}, BB::TupMat, 
                   nX = size(BB[1], 1)) where {NB}
   @assert all(B->size(B, 1) >= nX, BB)
   spec = basis.spec
   fill!(A, 0)
   @inbounds for (iA, ϕ) in enumerate(spec)
      a = zero(eltype(A))
      @simd ivdep for j = 1:nX
         b = ntuple(t -> BB[t][j, ϕ[t]], NB)
         a += @fastmath(prod(b))
      end
      A[iA] = a
   end

   return nothing
end




# struct LinearBatch
#    groups::Vector{Int}
# end

# function linearbatch(target::AbstractVector{<: Integer})
#    @assert issorted(target)
#    @assert minimum(target) > 0 
#    ngroups = target[end] 
#    groups = zeros(Int, ngroups+1)
#    gidx = 1 
#    i = 1
#    groups[1] = 1 
#    for gidx = 1:ngroups 
#       while (i <= length(target)) && (target[i] == gidx)
#          i += 1
#       end
#       groups[gidx+1] = i
#    end
#    return LinearBatch(groups)
# end

# evalpool_batch!(A, basis::PooledSparseProduct, BB, 
#                    target::AbstractVector{<: Integer}) = 
#     evalpool_batch!(A, basis, BB, linearbatch(target))

# function evalpool_batch!(A, basis::PooledSparseProduct{NB}, BB, 
#                          target::LinearBatch) where {NB}
#    nX = size(BB[1], 1)
#    nA = size(A, 1)
#    @assert length(target.groups)-1 <= nA 
#    @assert all(B->size(B, 1) == nX, BB)
#    spec = basis.spec

#    @inbounds for (iA, ϕ) in enumerate(spec)
#       for t = 1:length(target.groups)-1
#          a_t = zero(eltype(A))
#          @simd ivdep for j = target.groups[t]:target.groups[t+1]-1
#             a_t += BB_prod(ϕ, BB, j)
#          end
#          A[t, iA] = a_t 
#       end
#    end
#    return nothing
# end


# function evalpool!(A::VA, basis::PooledSparseProduct{2}, BB) where {VA}
#    nX = size(BB[1], 1)
#    @assert size(BB[2], 1) >= nX 
#    @assert length(A) == length(basis)
#    spec = basis.spec
#    BB1 = BB[1] 
#    BB2 = BB[2] 

#    @inbounds for (iA, ϕ) in enumerate(spec)
#       a = zero(eltype(A))
#       ϕ1 = ϕ[1]; ϕ2 = ϕ[2]
#       @simd ivdep for j = 1:nX
#          a = muladd(BB1[j, ϕ1], BB2[j, ϕ2], a)
#       end
#       A[iA] = a
#    end

#    return nothing
# end

# this code should never be used, we keep it just for testing 
# the performance of the generated code. 
# function prod_and_pool3!(A::VA, basis::PooledSparseProduct{3}, 
#                      BB::Tuple{TB1, TB2, TB3}) where {VA, TB1, TB2, TB3}
#    B1 = BB[1]; B2 = BB[2]; B3 = BB[3] 
#    # VT = promote_type(eltype(B1), eltype(B2), eltype(B3))
#    nX = size(B1, 1) 
#    @assert size(B2, 1) == size(B3, 1) == nX 
#    a = zero(eltype(A))

#    @inbounds for (iA, ϕ) in enumerate(basis.spec)
#       a *= 0
#       ϕ1 = ϕ[1]; ϕ2 = ϕ[2]; ϕ3 = ϕ[3]
#       @avx for j = 1:nX 
#          a += B1[j, ϕ1] * B2[j, ϕ2] * B3[j, ϕ3]
#       end
#       A[iA] = a
#    end
#    return A
# end


# -------------------- reverse mode gradient

using StaticArrays


function _rrule_evaluate(basis::PooledSparseProduct{NB}, BB::TupMat) where {NB}
   A = evaluate(basis, BB)
   return A, ∂A -> _pullback_evaluate(∂A, basis, BB)
end


function _pullback_evaluate(∂A, basis::PooledSparseProduct{NB}, BB::TupMat) where {NB}
   nX = size(BB[1], 1)
   TA = promote_type(eltype.(BB)..., eltype(∂A))
   ∂BB = ntuple(i -> zeros(TA, size(BB[i])...), NB)
   _pullback_evaluate!(∂BB, ∂A, basis, BB)
   return ∂BB
end

using Base.Cartesian: @nexprs 

function _pullback_evaluate!(∂BB, ∂A, basis::PooledSparseProduct{NB}, BB::TupMat) where {NB}
   nX = size(BB[1], 1)
   @assert all(nX <= size(BB[i], 1) for i = 1:NB)
   @assert all(nX <= size(∂BB[i], 1) for i = 1:NB)
   @assert all(size(∂BB[i], 2) >= size(BB[i], 2) for i = 1:NB)
   @assert length(∂A) == length(basis)
   @assert length(BB) == NB 
   @assert length(∂BB) == NB 
   
   @inbounds for (iA, ϕ) in enumerate(basis.spec)
      ∂A_iA = ∂A[iA]
      @simd ivdep for j = 1:nX 
         b = ntuple(Val(NB)) do i 
            @inbounds BB[i][j, ϕ[i]] 
         end 
         a, g = _static_prod_ed(b)
         for i = 1:NB 
            ∂BB[i][j, ϕ[i]] = muladd(∂A_iA, g[i], ∂BB[i][j, ϕ[i]])
         end
      end 
   end
   return nothing 
end

# TODO: interestingly the generic code above does not perform well 
#       in a production setting and we may want to return to 
#       a cruder code generation strategy. This specialized code 
#       confirms this. 

function _pullback_evaluate!(∂BB, ∂A, basis::PooledSparseProduct{2}, BB::TupMat)
   nX = size(BB[1], 1)
   NB = 2 
   @assert length(∂A) == length(basis)
   @assert length(BB) == length(∂BB) == 2
   @assert all(nX <= size(BB[i], 1) for i = 1:NB)
   @assert all(nX <= size(∂BB[i], 1) for i = 1:NB)
   @assert all(size(∂BB[i], 2) >= size(BB[i], 2) for i = 1:NB)
   
   @inbounds for (iA, ϕ) in enumerate(basis.spec)
      ∂A_iA = ∂A[iA]
      ϕ1 = ϕ[1]
      ϕ2 = ϕ[2]
      @simd ivdep for j = 1:nX 
         b1 = BB[1][j, ϕ1]
         b2 = BB[2][j, ϕ2]
         ∂BB[1][j, ϕ1] = muladd(∂A_iA, b2, ∂BB[1][j, ϕ1])
         ∂BB[2][j, ϕ2] = muladd(∂A_iA, b1, ∂BB[2][j, ϕ2])
      end 
   end
   return nothing 
end


function _pullback_evaluate!(∂BB, ∂A, basis::PooledSparseProduct{3}, BB::TupMat; 
                              sizecheck = true)
   nX = size(BB[1], 1)
   NB = 3 

   if sizecheck 
      @assert all(nX <= size(BB[i], 1) for i = 1:NB)
      @assert all(nX <= size(∂BB[i], 1) for i = 1:NB)
      @assert all(size(∂BB[i], 2) >= size(BB[i], 2) for i = 1:NB)
      @assert length(∂A) == length(basis)
      @assert length(BB) == NB 
      @assert length(∂BB) == NB 
   end

   B1 = BB[1]; B2 = BB[2]; B3 = BB[3]
   ∂B1 = ∂BB[1]; ∂B2 = ∂BB[2]; ∂B3 = ∂BB[3]
   
   @inbounds for (iA, ϕ) in enumerate(basis.spec)
      ∂A_iA = ∂A[iA]
      ϕ1 = ϕ[1]
      ϕ2 = ϕ[2]
      ϕ3 = ϕ[3]
      @simd ivdep for j = 1:nX 
         b1 = B1[j, ϕ1]
         b2 = B2[j, ϕ2]
         b3 = B3[j, ϕ3]
         ∂B1[j, ϕ1] = muladd(∂A_iA, b2*b3, ∂B1[j, ϕ1])
         ∂B2[j, ϕ2] = muladd(∂A_iA, b1*b3, ∂B2[j, ϕ2])
         ∂B3[j, ϕ3] = muladd(∂A_iA, b1*b2, ∂B3[j, ϕ3])
      end 
   end
   return nothing 
end

import ForwardDiff

function _pb_pb_evaluate(basis::PooledSparseProduct{NB}, ∂2, 
                         ∂A, BB::TupMat) where {NB}

   # ∂2 should be a tuple of length 2
   @assert ∂2 isa NTuple{NB, <: AbstractMatrix}
   @assert BB isa NTuple{NB,  <: AbstractMatrix}
   @assert ∂A isa AbstractVector

   nX = size(BB[1], 1)
   @assert all(nX == size(BB[i], 1) for i = 1:NB)

   ∂2_∂A = zeros(length(∂A))
   ∂2_BB = ntuple(i -> zeros(size(BB[i])...), NB)

   for (iA, ϕ) in enumerate(basis.spec)
      @simd ivdep for j = 1:nX 
         b = ntuple(Val(NB)) do i 
            @inbounds BB[i][j, ϕ[i]] 
         end 
         ∂g = ntuple(Val(NB)) do i 
            @inbounds ∂2[i][j, ϕ[i]]
         end
         _, g, ∂g_b = _pb_grad_static_prod(∂g, b)
         for i = 1:NB 
            # ∂BB[i][j, ϕ[i]] += ∂A[iA] * g[i]
            ∂2_∂A[iA] += ∂2[i][j, ϕ[i]] * g[i]
            ∂2_BB[i][j, ϕ[i]] += ∂A[iA] * ∂g_b[i]
         end
      end 
   end
   return ∂2_∂A, ∂2_BB 
end


function _pb_pb_evaluate(basis::PooledSparseProduct{1}, ∂2, 
                         ∂A, BB::TupMat)

   # ∂2 should be a tuple of length 2
   @assert ∂2 isa Tuple{<: AbstractMatrix}
   @assert BB isa Tuple{<: AbstractMatrix}
   @assert ∂A isa AbstractVector
   
   nX = size(BB[1], 1)

   ∂2_∂A = zeros(length(∂A))
   ∂2_BB = (zeros(size(BB[1])...), )
   
   for (iA, ϕ) in enumerate(basis.spec)
      @simd ivdep for j = 1:nX 
         ϕ1 = ϕ[1]
         b1 = BB[1][j, ϕ1]
         # A[iA] += b1
         # ∂BB[1][j, ϕ1] += ∂A[iA]
         ∂2_∂A[iA] += ∂2[1][j, ϕ1] 
      end 
   end
   return ∂2_∂A, ∂2_BB 
end


function _pb_pb_evaluate(basis::PooledSparseProduct{2}, ∂2, 
                         ∂A, BB::TupMat)

   # ∂2 should be a tuple of length 2
   @assert ∂2 isa Tuple{<: AbstractMatrix, <: AbstractMatrix}
   @assert BB isa Tuple{<: AbstractMatrix, <: AbstractMatrix}
   @assert ∂A isa AbstractVector
   
   nX = size(BB[1], 1)

   ∂2_∂A = zeros(length(∂A))
   ∂2_BB = ntuple(i -> zeros(size(BB[i])...), 2)
   
   for (iA, ϕ) in enumerate(basis.spec)
      @simd ivdep for j = 1:nX 
         ϕ1 = ϕ[1]
         ϕ2 = ϕ[2]
         b1 = BB[1][j, ϕ1]
         b2 = BB[2][j, ϕ2]
         # A[iA] += b1 * b2 
         # ∂BB[1][j, ϕ1] += ∂A[iA] * b2
         # ∂BB[2][j, ϕ2] += ∂A[iA] * b1
         ∂2_∂A[iA] += ∂2[1][j, ϕ1] * b2 + ∂2[2][j, ϕ2] * b1
         ∂2_BB[1][j, ϕ1] += ∂2[2][j, ϕ2] * ∂A[iA]
         ∂2_BB[2][j, ϕ2] += ∂2[1][j, ϕ1] * ∂A[iA]
      end 
   end
   return ∂2_∂A, ∂2_BB 
end




# function _pullback_evaluate!(∂BB, ∂A, basis::PooledSparseProduct{NB}, 
#                              BB::Tuple, target::AbstractVector{<: Integer}) where {NB}
#    nX = size(BB[1], 1)
#    nT = size(∂A, 1)
#    mint, maxt = extrema(target)
#    @assert 0 < mint <= maxt <= nT
#    @assert all(nX <= size(BB[i], 1) for i = 1:NB)
#    @assert all(nX <= size(∂BB[i], 1) for i = 1:NB)
#    @assert all(size(∂BB[i], 2) >= size(BB[i], 2) for i = 1:NB)
#    @assert size(∂A, 2) == length(basis)
#    @assert length(BB) == NB 
#    @assert length(∂BB) == NB 

#    # ∂A_loc = zeros(eltype(∂A), nT)

#    @inbounds for (iA, ϕ) in enumerate(basis.spec)
#       # @simd ivdep for t = 1:nT 
#       #    ∂A_loc[t] = ∂A[t, iA]
#       # end
#       @simd ivdep for j = 1:nX 
#          ∂A_iA = ∂A[target[j], iA] # ∂A_loc[target[j] ] 
#          b = ntuple(Val(NB)) do i 
#             @inbounds BB[i][j, ϕ[i]] 
#          end 
#          g = _prod_grad(b, Val(NB))
#          for i = 1:NB 
#             ∂BB[i][j, ϕ[i]] = muladd(∂A_iA, g[i], ∂BB[i][j, ϕ[i]])
#          end
#       end 
#    end
#    return nothing 
# end


# --------------------- connect with ChainRules 
# todo ... 

import ChainRulesCore: rrule, NoTangent

function rrule(::typeof(evaluate), basis::PooledSparseProduct{NB}, BB::TupMat) where {NB}
   A = evaluate(basis, BB)

   function pb(Δ)
      ∂BB = _pullback_evaluate(Δ, basis, BB)
      return NoTangent(), NoTangent(), ∂BB
   end 

   return A, pb 
end

function rrule(::typeof(_pullback_evaluate), Δ, basis::PooledSparseProduct, BB)
   ∂BB = _pullback_evaluate(Δ, basis, BB)

   function pb(Δ2)
      ∂2_Δ, ∂2_BB = _pb_pb_evaluate(basis, Δ2, Δ, BB)
      return NoTangent(), ∂2_Δ, NoTangent(), ∂2_BB
   end

   return ∂BB, pb
end




# --------------------- connect with Lux 
# it looks like we could use the standard P4ML basis wrapper 
# but technically the pooling operation changes the behaviour in
# a few ways and we need to be very careful about this

import LuxCore: AbstractExplicitLayer, initialparameters, initialstates

struct PooledSparseProductLayer{NB} <: AbstractExplicitLayer 
   basis::PooledSparseProduct{NB}
end

lux(basis::PooledSparseProduct) = PooledSparseProductLayer(basis)

initialparameters(rng::AbstractRNG, layer::PooledSparseProductLayer) = 
      NamedTuple() 

initialstates(rng::AbstractRNG, layer::PooledSparseProductLayer) = 
      NamedTuple()

(l::PooledSparseProductLayer)(BB, ps, st) = 
      evaluate(l.basis, BB), st 
