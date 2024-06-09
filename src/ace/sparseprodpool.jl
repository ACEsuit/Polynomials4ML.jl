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
struct PooledSparseProduct{NB} <: AbstractP4MLTensor
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


# ----------------------- evaluation and allocation interfaces 

_valtype(basis::PooledSparseProduct, BB::Tuple) = 
      mapreduce(eltype, promote_type, BB)

_gradtype(basis::PooledSparseProduct, BB::Tuple) = 
      mapreduce(eltype, promote_type, BB)

function _generate_input_1(basis::PooledSparseProduct{NB}) where {NB} 
   NN = [ maximum(b[i] for b in basis.spec) for i = 1:NB ]
   BB = ntuple(i -> randn(NN[i]), NB)
   return BB 
end 

function _generate_input(basis::PooledSparseProduct{NB}; nX = rand(5:15)) where {NB} 
   NN = [ maximum(b[i] for b in basis.spec) for i = 1:NB ]
   BB = ntuple(i -> randn(nX, NN[i]), NB)
   return BB 
end 

function _generate_batch(basis::PooledSparseProduct, args...; kwargs...) 
   error("PooledSparseProduct is not implemented for batch inputs")
end


# ----------------------- evaluation kernels 

# Valentin Churavy's Version (which we don't really understand)
#
# # Stolen from KernelAbstractions
# import Base.Cartesian: @nexprs
# import Adapt 
# struct ConstAdaptor end
# import Base.Experimental: @aliasscope
# Adapt.adapt_storage(::ConstAdaptor, a::Array) = Base.Experimental.Const(a)
# constify(arg) = Adapt.adapt(ConstAdaptor(), arg)
# 
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

function whatalloc(evaluate!, basis::PooledSparseProduct{NB}, BB::TupVecMat) where {NB}
   TV = _valtype(basis, BB)
   nA = length(basis)
   return (TV, nA) 
end


function evaluate!(A, basis::PooledSparseProduct{NB}, BB::TupVec) where {NB}
   spec = basis.spec
   @assert length(A) >= length(spec)
   fill!(A, 0)
   @inbounds for (iA, ϕ) in enumerate(spec)
      b = ntuple(t -> BB[t][ϕ[t]], NB)
      A[iA] = @fastmath(prod(b))
   end
   return A
end


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
   return A
end

# special-casing NB = 2 for performance reasons
function evaluate!(A, basis::PooledSparseProduct{2}, 
                   BB::Tuple{<: AbstractMatrix, <: AbstractMatrix},
                   nX = size(BB[1], 1))
   @assert all(B->size(B, 1) >= nX, BB)
   BB1, BB2 = BB
   spec = basis.spec
   fill!(A, zero(eltype(A)))
   @inbounds for (iA, ϕ) in enumerate(spec)
      ϕ1, ϕ2 = ϕ
      a = zero(eltype(A))
      @simd ivdep for j = 1:nX
         b1 = BB1[j, ϕ1]
         b2 = BB2[j, ϕ2]
         a = muladd(b1, b2, a)
      end
      A[iA] = a
   end
   return A
end



# -------------------- reverse mode gradient

using StaticArrays


function whatalloc(::typeof(pullback_evaluate!), 
                   ∂A, basis::PooledSparseProduct{NB}, BB::TupMat) where  {NB}
   TA = promote_type(eltype.(BB)..., eltype(∂A))
   return ntuple(i -> (TA, size(BB[i])...), NB)                   
end


# the next few method definitions ensure that we can use the 
# WithAlloc stuff with the pullback_evaluate! function.
# TODO: this should probably be replaced with a loop that generates  
# the code up to a large-ish NB. 

pullback_evaluate!(∂B1, ∂A, basis::PooledSparseProduct{1}, BB::TupMat) = 
         pullback_evaluate!((∂B1,), ∂A, basis, BB)

pullback_evaluate!(∂B1, ∂B2, ∂A, basis::PooledSparseProduct{2}, BB::TupMat) = 
         pullback_evaluate!((∂B1, ∂B2,), ∂A, basis, BB)

pullback_evaluate!(∂B1, ∂B2, ∂B3, ∂A, basis::PooledSparseProduct{3}, BB::TupMat) = 
         pullback_evaluate!((∂B1, ∂B2, ∂B3,), ∂A, basis, BB)

pullback_evaluate!(∂B1, ∂B2, ∂B3, ∂B4, ∂A, basis::PooledSparseProduct{4}, BB::TupMat) = 
         pullback_evaluate!((∂B1, ∂B2, ∂B3, ∂B4,), ∂A, basis, BB)


function pullback_evaluate!(∂BB, # output 
                            ∂A, basis::PooledSparseProduct{NB}, BB::TupMat # inputs 
                            ) where {NB}
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
            BB[i][j, ϕ[i]] 
         end 
         a, g = _static_prod_ed(b)
         for i = 1:NB 
            ∂BB[i][j, ϕ[i]] = muladd(∂A_iA, g[i], ∂BB[i][j, ϕ[i]])
         end
      end 
   end
   return ∂BB 
end

# TODO: interestingly the generic code above does not perform well 
#       in a production setting and we may want to return to 
#       a cruder code generation strategy. This specialized code 
#       confirms this. 

function pullback_evaluate!(∂BB, ∂A, basis::PooledSparseProduct{2}, BB::TupMat)
   nX = size(BB[1], 1)
   NB = 2 
   @assert length(∂A) == length(basis)
   @assert length(BB) == length(∂BB) == 2
   @assert all(nX <= size(BB[i], 1) for i = 1:NB)
   @assert all(nX <= size(∂BB[i], 1) for i = 1:NB)
   @assert all(size(∂BB[i], 2) >= size(BB[i], 2) for i = 1:NB)
   BB1, BB2 = BB
   ∂BB1, ∂BB2 = ∂BB
   
   @inbounds for (iA, ϕ) in enumerate(basis.spec)
      ∂A_iA = ∂A[iA]
      ϕ1 = ϕ[1]
      ϕ2 = ϕ[2]
      @simd ivdep for j = 1:nX 
         b1 = BB1[j, ϕ1]
         b2 = BB2[j, ϕ2]
         ∂BB1[j, ϕ1] = muladd(∂A_iA, b2, ∂BB1[j, ϕ1])
         ∂BB2[j, ϕ2] = muladd(∂A_iA, b1, ∂BB2[j, ϕ2])
      end 
   end
   return ∂BB 
end


function pullback_evaluate!(∂BB, ∂A, basis::PooledSparseProduct{3}, BB::TupMat; 
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
   return ∂BB 
end


# --------------------------------------------------------
#  reverse over reverse 

function pb_pb_evaluate(∂2, ∂A, basis::PooledSparseProduct{NB}, BB::TupMat
                        ) where {NB}

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


function pb_pb_evaluate(∂2, ∂A, basis::PooledSparseProduct{1}, BB::TupMat)

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


function pb_pb_evaluate(∂2, ∂A, basis::PooledSparseProduct{2}, BB::TupMat)

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


# --------------------- Pushforwards 

# This implementation of the pushforward doesn't yet do batching 
# this means that the output is a single vector A, the inputs 
#        BB[i] are nX x nBi -> with the nX pooled 
# It is ASSUMED that ∂BB[i][j, :] / ∂X[j'] = 0 if j ≠ j'
# Therefore    ΔBB[i] are also nX x nBi
# This is a simplification that may have to be revisited at some point. 
#
# The output will be  A, ∂A   where 
#       A is size (nA,)
#      ∂A is size (nA, nX)

_my_promote_type(args...) = promote_type(args...)

_my_promote_type(T1::Type{<: Number}, T2::Type{SVector{N, S}}, args...
                  ) where {N, S} = 
      promote_type(SVector{N, T1}, T2, args...)



function pushforward_evaluate(basis::PooledSparseProduct, BB, ΔBB)
   @assert length(size(BB[1])) == 2
   @assert length(size(ΔBB[1])) == 2
   @assert all(size(BB[t]) == size(ΔBB[t]) for t = 1:length(BB))
   
   nX = size(ΔBB[1], 1)
   nA = length(basis)
   
   TA = promote_type(eltype.(BB)...)
   A = zeros(TA, nA)
   
   T∂A = _my_promote_type(TA, eltype.(ΔBB)...)
   ∂A = zeros(T∂A, (nA, nX))
   fill!(∂A, zero(T∂A))

   return pushforward_evaluate!(A, ∂A, basis, BB, ΔBB)
end   


function pushforward_evaluate!(A, ∂A, basis::PooledSparseProduct{NB}, BB, ΔBB) where {NB}
   nX = size(BB[1], 1)
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



# --------------------- connect with ChainRules 
# can this be generalized again? 

import ChainRulesCore: rrule, NoTangent

function rrule(::typeof(evaluate), basis::PooledSparseProduct{NB}, BB::TupMat) where {NB}
   A = evaluate(basis, BB)

   function pb(Δ)
      ∂BB = pullback_evaluate(Δ, basis, BB)
      return NoTangent(), NoTangent(), ∂BB
   end 

   return A, pb 
end


function rrule(::typeof(pullback_evaluate), Δ, basis::PooledSparseProduct, BB)
   ∂BB = pullback_evaluate(Δ, basis, BB)

   function pb(Δ2)
      ∂2_Δ, ∂2_BB = pb_pb_evaluate(Δ2, Δ, basis, BB)
      return NoTangent(), ∂2_Δ, NoTangent(), ∂2_BB
   end

   return ∂BB, pb
end

# TODO: frules 



# --------------------- connect with Lux 
# it looks like we could use the standard P4ML basis wrapper 
# but technically the pooling operation changes the behaviour in
# a few ways and we need to be very careful about this

import LuxCore: AbstractExplicitLayer, initialparameters, initialstates

struct PooledSparseProductLayer{NB} <: AbstractExplicitLayer 
   basis::PooledSparseProduct{NB}
   meta::Dict{String, Any}
   release_input::Bool
end

function lux(basis::PooledSparseProduct; 
               name = String(nameof(typeof(basis))), 
               meta = Dict{String, Any}("name" => name),
               release_input = true)
   @assert haskey(meta, "name")
   return PooledSparseProductLayer(basis, meta, release_input)
end

initialparameters(rng::AbstractRNG, layer::PooledSparseProductLayer) = 
      NamedTuple() 

initialstates(rng::AbstractRNG, layer::PooledSparseProductLayer) = 
      NamedTuple()

(l::PooledSparseProductLayer)(BB, ps, st) = begin
   out = evaluate(l.basis, BB)
   if l.release_input
      release!.(BB)
   end
   return out, st
end
