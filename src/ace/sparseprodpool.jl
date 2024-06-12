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

# special-casing NB = 1 for correctness 
function evaluate!(A, basis::PooledSparseProduct{1}, 
                   BB::Tuple{<: AbstractMatrix},
                   nX = size(BB[1], 1))
   @assert size(BB[1], 1) >= nX
   BB1 = BB[1] 
   spec = basis.spec
   fill!(A, zero(eltype(A)))
   @inbounds for (iA, ϕ) in enumerate(spec)
      ϕ1 = ϕ[1]
      a = zero(eltype(A))
      @simd ivdep for j = 1:nX
         b1 = BB1[j, ϕ1]
         a += b1
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


function whatalloc(::typeof(pullback!), 
                   ∂A, basis::PooledSparseProduct{NB}, BB::TupMat) where  {NB}
   TA = promote_type(eltype.(BB)..., eltype(∂A))
   return ntuple(i -> (TA, size(BB[i])...), NB)                   
end


# the next few method definitions ensure that we can use the 
# WithAlloc stuff with the pullback! function.
# TODO: this should probably be replaced with a loop that generates  
# the code up to a large-ish NB. 

pullback!(∂B1::AbstractMatrix, ∂A, basis::PooledSparseProduct{1}, BB::TupMat) = 
         pullback!((∂B1,), ∂A, basis, BB)

pullback!(∂B1, ∂B2, ∂A, basis::PooledSparseProduct{2}, BB::TupMat) = 
         pullback!((∂B1, ∂B2,), ∂A, basis, BB)

pullback!(∂B1, ∂B2, ∂B3, ∂A, basis::PooledSparseProduct{3}, BB::TupMat) = 
         pullback!((∂B1, ∂B2, ∂B3,), ∂A, basis, BB)

pullback!(∂B1, ∂B2, ∂B3, ∂B4, ∂A, basis::PooledSparseProduct{4}, BB::TupMat) = 
         pullback!((∂B1, ∂B2, ∂B3, ∂B4,), ∂A, basis, BB)


function pullback!(∂BB::Tuple, # output 
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
            ϕi = ϕ[i]
            ∂BB[i][j, ϕi] = muladd(∂A_iA, g[i], ∂BB[i][j, ϕi])
         end
      end 
   end
   return ∂BB 
end

# TODO: interestingly the generic code above does not perform well 
#       in a production setting and we may want to return to 
#       a cruder code generation strategy. This specialized code 
#       confirms this. 

# NB = 1 for correctness 
function pullback!(∂BB::Tuple, ∂A, basis::PooledSparseProduct{1}, BB::TupMat)
   nX = size(BB[1], 1)
   NB = 1
   @assert length(∂A) == length(basis)
   @assert length(BB) == length(∂BB) == NB 
   @assert all(nX <= size(BB[i], 1) for i = 1:NB)
   @assert all(nX <= size(∂BB[i], 1) for i = 1:NB)
   @assert all(size(∂BB[i], 2) >= size(BB[i], 2) for i = 1:NB)
   BB1 = BB[1]
   ∂BB1 = ∂BB[1]

   fill!(∂BB1, zero(eltype(∂BB1)))
   
   @inbounds for (iA, ϕ) in enumerate(basis.spec)
      ∂A_iA = ∂A[iA]
      ϕ1 = ϕ[1]
      @simd ivdep for j = 1:nX 
         # A[iA] += b1 
         ∂BB1[j, ϕ1] += ∂A_iA
      end 
   end
   return ∂BB 
end

function pullback!(∂BB::Tuple, ∂A, basis::PooledSparseProduct{2}, BB::TupMat)
   nX = size(BB[1], 1)
   NB = 2 
   @assert length(∂A) == length(basis)
   @assert length(BB) == length(∂BB) == 2
   @assert all(nX <= size(BB[i], 1) for i = 1:NB)
   @assert all(nX <= size(∂BB[i], 1) for i = 1:NB)
   @assert all(size(∂BB[i], 2) >= size(BB[i], 2) for i = 1:NB)
   BB1, BB2 = BB
   ∂BB1, ∂BB2 = ∂BB

   for i = 1:length(∂BB)
      fill!(∂BB[i], zero(eltype(∂BB[i])))
   end
   
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

function pullback!(∂BB::Tuple, ∂A, basis::PooledSparseProduct{3}, BB::TupMat; 
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

   for i = 1:length(∂BB)
      fill!(∂BB[i], zero(eltype(∂BB[i])))
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

function pullback!(∂BB::Tuple, ∂A, basis::PooledSparseProduct{4}, BB::TupMat; 
                              sizecheck = true)
   nX = size(BB[1], 1)
   NB = 4 

   if sizecheck 
      @assert all(nX <= size(BB[i], 1) for i = 1:NB)
      @assert all(nX <= size(∂BB[i], 1) for i = 1:NB)
      @assert all(size(∂BB[i], 2) >= size(BB[i], 2) for i = 1:NB)
      @assert length(∂A) == length(basis)
      @assert length(BB) == NB 
      @assert length(∂BB) == NB 
   end

   for i = 1:length(∂BB)
      fill!(∂BB[i], zero(eltype(∂BB[i])))
   end
   
   B1 = BB[1]; B2 = BB[2]; B3 = BB[3]; B4 = BB[4]
   ∂B1 = ∂BB[1]; ∂B2 = ∂BB[2]; ∂B3 = ∂BB[3]; ∂B4 = ∂BB[4]
   
   @inbounds for (iA, ϕ) in enumerate(basis.spec)
      ∂A_iA = ∂A[iA]
      ϕ1 = ϕ[1]
      ϕ2 = ϕ[2]
      ϕ3 = ϕ[3]
      ϕ4 = ϕ[4]
      @simd ivdep for j = 1:nX 
         b1 = B1[j, ϕ1]
         b2 = B2[j, ϕ2]
         b3 = B3[j, ϕ3]
         b4 = B4[j, ϕ4]
         ∂B1[j, ϕ1] = muladd(∂A_iA, b2*b3*b4, ∂B1[j, ϕ1])
         ∂B2[j, ϕ2] = muladd(∂A_iA, b1*b3*b4, ∂B2[j, ϕ2])
         ∂B3[j, ϕ3] = muladd(∂A_iA, b1*b2*b4, ∂B3[j, ϕ3])
         ∂B4[j, ϕ4] = muladd(∂A_iA, b1*b2*b3, ∂B4[j, ϕ4])
      end 
   end
   return ∂BB 
end


# ---------------------------------------------------------------
#  reverse over reverse 

#    A = evaluate(basis, BB)
#  ∂BB = pullback(∂A, basis, BB) 
# ∂∂BB is the perturbation to ∂BB

function whatalloc(::typeof(pullback2!), ∂∂BB, ∂A, 
                   basis::PooledSparseProduct{NB}, BB) where {NB}
   TA = promote_type(eltype.(BB)..., eltype(∂A), eltype.(∂∂BB)...)
   return ( (TA, size(∂A)...), 
            ntuple(i -> (TA, size(BB[i])...), NB)...)
end

pullback2!(∇_∂A, ∇_BB1::AbstractMatrix, ∂∂BB, ∂A, basis::PooledSparseProduct{1}, BB) = 
      pullback2!(∇_∂A, (∇_BB1,), ∂∂BB, ∂A, basis, BB) 

pullback2!(∇_∂A, ∇_BB1, ∇_BB2, ∂∂BB, ∂A, basis::PooledSparseProduct{2}, BB) = 
      pullback2!(∇_∂A, (∇_BB1, ∇_BB2), ∂∂BB, ∂A, basis, BB) 

pullback2!(∇_∂A, ∇_BB1, ∇_BB2, ∇_BB3, ∂∂BB, ∂A, basis::PooledSparseProduct{3}, BB) = 
      pullback2!(∇_∂A, (∇_BB1, ∇_BB2, ∇_BB3), ∂∂BB, ∂A, basis, BB) 

pullback2!(∇_∂A, ∇_BB1, ∇_BB2, ∇_BB3, ∇_BB4, ∂∂BB, ∂A, basis::PooledSparseProduct{4}, BB) = 
      pullback2!(∇_∂A, (∇_BB1, ∇_BB2, ∇_BB3, ∇_BB4), ∂∂BB, ∂A, basis, BB) 


function pullback2!(∇_∂A, ∇_BB::Tuple,  # outputs 
                    ∂∂BB,    # perturbation 
                    ∂A, basis::PooledSparseProduct{NB}, BB)  where {NB}

   function _dual(i)
      T = promote_type(eltype(BB[i]), eltype(∂∂BB[i]))
      return Dual{T}(zero(T), one(T))
   end

   @no_escape begin 
      dd = ntuple(_dual, NB)
      BB_d = ntuple(i -> @alloc(typeof(dd[i]), size(BB[i])...), NB)
      for i = 1:NB
         @inbounds for t = 1:length(BB[i])
            BB_d[i][t] = BB[i][t] + dd[i] * ∂∂BB[i][t]
         end
      end
      A_d = @withalloc evaluate!(basis, BB_d)
      ∂BB_d = @withalloc pullback!(∂A, basis, BB_d)
      @inbounds for t = 1:length(A_d)
         ∇_∂A[t] = extract_derivative(eltype(∇_∂A), A_d[t])
      end
      @inbounds for i = 1:NB
         for t = 1:length(∂BB_d[i])
            Ti = eltype(∇_BB[i])
            ∇_BB[i][t] = extract_derivative(Ti, ∂BB_d[i][t])
         end
      end
   end
   return ∇_∂A, ∇_BB
end


# ---------------------------------------------------------------
#  Pushforward  

using ForwardDiff: value

function whatalloc(::typeof(pushforward!), 
                   basis::PooledSparseProduct{NB}, BB, ∂BB) where {NB}
   TA = promote_type(eltype.(BB)...) 
   T∂A = promote_type(TA, eltype.(∂BB)...)
   return (TA, length(basis)), (T∂A, length(basis))
end

function pushforward!(A, ∂A, basis::PooledSparseProduct{NB}, BB, ∂BB) where {NB}

   function _dual(i)
      T = promote_type(eltype(BB[i]), eltype(∂BB[i]))
      return Dual{T}(zero(T), one(T))
   end

   @no_escape begin 
      dd = ntuple(_dual, NB)
      BB_d = ntuple(i -> @alloc(typeof(dd[i]), size(BB[i])...), NB)
      for i = 1:NB
         @inbounds for t = 1:length(BB[i])
            BB_d[i][t] = BB[i][t] + dd[i] * ∂BB[i][t]
         end
      end
      A_d = @withalloc evaluate!(basis, BB_d)
      for t = 1:length(A_d)
         A[t] = value(eltype(A), A_d[t])
         ∂A[t] = extract_derivative(eltype(∂A), A_d[t])
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
      ∂BB = pullback(Δ, basis, BB)
      return NoTangent(), NoTangent(), ∂BB
   end 

   return A, pb 
end


function rrule(::typeof(pullback), Δ, basis::PooledSparseProduct, BB)
   ∂BB = pullback(Δ, basis, BB)

   function pb(Δ2)
      ∂2_Δ, ∂2_BB = pullback2(Δ2, Δ, basis, BB)
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
