

struct PooledSparseProduct{NB}
   spec::Vector{NTuple{NB, Int}}
   # ---- temporaries & caches 
end

function PooledSparseProduct()
   return PooledSparseProduct(bases, NTuple{NB, Int}[])
end

# each column defines a basis element
function PooledSparseProduct(spec::Matrix{<: Integer})
   @assert all(spec .> 0)
   spect = [ Tuple(spec[:, i]...) for i = 1:size(spec, 2) ]
   return PooledSparseProduct(spect)
end

Base.length(basis::PooledSparseProduct) = length(basis.spec)

# function Base.show(io::IO, basis::PooledSparseProduct)
#    print(io, "PooledSparseProduct(")
#    print(io, basis.bases)
# end


# ----------------------- evaluation interfaces 


function evaluate(basis::PooledSparseProduct, BB::Tuple) 
   VT = mapreduce(eltype, promote_type, BB)
   A = zeros(VT, length(basis))
   evaluate!(A, basis, BB::Tuple)
   return A 
end

function evalpool(basis::PooledSparseProduct, BB::Tuple)
   VT = mapreduce(eltype, promote_type, BB)
   A = zeros(VT, length(basis))
   evalpool!(A, basis, BB::Tuple)
   return A
end


test_evaluate(basis::PooledSparseProduct, BB::Tuple) = 
       [ prod(BB[j][basis.spec[i][j]] for j = 1:length(BB)) 
            for i = 1:length(basis) ]

test_evalpool(basis::PooledSparseProduct, BB::Tuple) = 
      sum( test_evaluate(basis, ntuple(i -> BB[i][j, :], length(BB)))
         for j = 1:size(BB[1], 1) )            

# ----------------------- evaluation kernels 

import Base.Cartesian: @nexprs

# Stolen from KernelAbstractions
import Adapt 
struct ConstAdaptor end

import Base.Experimental: @aliasscope

Adapt.adapt_storage(::ConstAdaptor, a::Array) = Base.Experimental.Const(a)
constify(arg) = Adapt.adapt(ConstAdaptor(), arg)


@inline function BB_prod(ϕ::NTuple{NB}, BB) where NB
   reduce(Base.FastMath.mul_fast, ntuple(Val(NB)) do i
      @inline 
      @inbounds BB[i][ϕ[i]]
   end)
end


function evaluate!(A, basis::PooledSparseProduct{NB}, BB) where {NB}
   @assert length(BB) == NB
   # evaluate the 1p product basis functions and add/write into _A
   BB = constify(BB)
   spec = constify(basis.spec)
   @aliasscope begin # No store to A aliases any read from any B
      for (iA, ϕ) in enumerate(spec)
         @inbounds A[iA] += BB_prod(ϕ, BB)
      end
   end
   return nothing 
end

@inline function BB_prod(ϕ::NTuple{NB}, BB, j) where NB
   reduce(Base.FastMath.mul_fast, ntuple(Val(NB)) do i
      @inline 
      @inbounds BB[i][j, ϕ[i]]
   end)
end


# BB::tuple of matrices 
function evalpool!(A, basis::PooledSparseProduct{NB}, BB, 
                   nX = size(BB[1], 1)) where {NB}
   @assert all(B->size(B, 1) >= nX, BB)
   BB = constify(BB) # Assumes that no B aliases A
   spec = constify(basis.spec)

   @aliasscope begin # No store to A aliases any read from any B
      @inbounds for (iA, ϕ) in enumerate(spec)
         a = zero(eltype(A))
         @simd ivdep for j = 1:nX
            a += BB_prod(ϕ, BB, j)
         end
         A[iA] = a
      end
   end
   return nothing
end


struct LinearBatch
   groups::Vector{Int}
end

function linearbatch(target::AbstractVector{<: Integer})
   @assert issorted(target)
   @assert minimum(target) > 0 
   ngroups = target[end] 
   groups = zeros(Int, ngroups+1)
   gidx = 1 
   i = 1
   groups[1] = 1 
   for gidx = 1:ngroups 
      while (i <= length(target)) && (target[i] == gidx)
         i += 1
      end
      groups[gidx+1] = i
   end
   return LinearBatch(groups)
end

evalpool_batch!(A, basis::PooledSparseProduct, BB, 
                   target::AbstractVector{<: Integer}) = 
    evalpool_batch!(A, basis, BB, linearbatch(target))

function evalpool_batch!(A, basis::PooledSparseProduct{NB}, BB, 
                         target::LinearBatch) where {NB}
   nX = size(BB[1], 1)
   nA = size(A, 1)
   @assert length(target.groups)-1 <= nA 
   @assert all(B->size(B, 1) == nX, BB)
   BB = constify(BB) # Assumes that no B aliases A
   spec = constify(basis.spec)

   @inbounds for (iA, ϕ) in enumerate(spec)
      for t = 1:length(target.groups)-1
         a_t = zero(eltype(A))
         @simd ivdep for j = target.groups[t]:target.groups[t+1]-1
            a_t += BB_prod(ϕ, BB, j)
         end
         A[t, iA] = a_t 
      end
   end
   return nothing
end


function evalpool!(A::VA, basis::PooledSparseProduct{2}, BB) where {VA}
   nX = size(BB[1], 1)
   @assert size(BB[2], 1) >= nX 
   @assert length(A) == length(basis)
   BB = constify(BB) # Assumes that no B aliases A
   spec = constify(basis.spec)
   BB1 = BB[1] 
   BB2 = BB[2] 

   @inbounds for (iA, ϕ) in enumerate(spec)
      a = zero(eltype(A))
      ϕ1 = ϕ[1]; ϕ2 = ϕ[2]
      @simd ivdep for j = 1:nX
         a = muladd(BB1[j, ϕ1], BB2[j, ϕ2], a)
      end
      A[iA] = a
   end

   return nothing
end

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


@inline function _prod_grad(b, ::Val{1})
   return (one(eltype(b)),)
end


@inline function _prod_grad(b::SVector{1, T}) where {T} 
   return b[1], SVector(one(T))
end


function _code_prod_grad(NB)
   code = Expr[] 
   # g[2] = b[1] 
   push!(code, :(g2 = b[1]))
   for i = 3:NB 
      # g[i] = g[i-1] * b[i-1]
      push!(code, Meta.parse("g$i = g$(i-1) * b[$(i-1)]"))
   end
   # h = b[N]
   push!(code, Meta.parse("h = b[$NB]"))
   for i = NB-1:-1:2
      # g[i] *= h
      push!(code, Meta.parse("g$i *= h"))
      # h *= b[i]
      push!(code, Meta.parse("h *= b[$i]"))
   end
   # g[1] = h
   push!(code, :(g1 = h))
   # return (g[1], g[2], ..., g[N])
   push!(code, Meta.parse(
            "return (" * join([ "g$i" for i = 1:NB ], ", ") * ")" ))
end

@inline @generated function _prod_grad(b, ::Val{NB}) where {NB}
   code = _code_prod_grad(NB)
   quote
      @fastmath begin 
         $(code...)
      end
   end
end



function _rrule_evalpool(basis::PooledSparseProduct{NB}, BB::Tuple) where {NB}
   A = evalpool(basis, BB)
   return A, ∂A -> _pullback_evalpool(∂A, basis, BB)
end


function _pullback_evalpool(∂A, basis::PooledSparseProduct{NB}, BB::Tuple) where {NB}
   nX = size(BB[1], 1)
   TA = promote_type(eltype.(BB)...)
   ∂BB = ntuple(i -> zeros(TA, size(BB[i])...), NB)
   _pullback_evalpool!(∂BB, ∂A, basis, BB)
   return ∂BB
end


function _pullback_evalpool!(∂BB, ∂A, basis::PooledSparseProduct{NB}, BB::Tuple) where {NB}
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
         g = _prod_grad(b, Val(NB))
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

function _pullback_evalpool!(∂BB, ∂A, basis::PooledSparseProduct{2}, BB::Tuple)
   nX = size(BB[1], 1)
   NB = 2 
   @assert length(∂A) == length(basis)
   @assert length(BB) == length(∂BB) == 2
   @assert all(nX <= size(BB[i], 1) for i = 1:NB)
   @assert all(nX <= size(∂BB[i], 1) for i = 1:NB)
   @assert all(size(∂BB[i], 2) >= size(BB[i], 2) for i = 1:NB)
   
   @inbounds for (iA, ϕ) in enumerate(basis.spec)
      ∂A_iA = ∂A[iA]
      @simd ivdep for j = 1:nX 
         ϕ1 = ϕ[1]
         ϕ2 = ϕ[2]
         b1 = BB[1][j, ϕ1]
         b2 = BB[2][j, ϕ2]
         ∂BB[1][j, ϕ1] = muladd(∂A_iA, b2, ∂BB[1][j, ϕ1])
         ∂BB[2][j, ϕ2] = muladd(∂A_iA, b1, ∂BB[2][j, ϕ2])
      end 
   end
   return nothing 
end



function _pullback_evalpool!(∂BB, ∂A, basis::PooledSparseProduct{NB}, 
                             BB::Tuple, target::AbstractVector{<: Integer}) where {NB}
   nX = size(BB[1], 1)
   nT = size(∂A, 1)
   mint, maxt = extrema(target)
   @assert 0 < mint <= maxt <= nT
   @assert all(nX <= size(BB[i], 1) for i = 1:NB)
   @assert all(nX <= size(∂BB[i], 1) for i = 1:NB)
   @assert all(size(∂BB[i], 2) >= size(BB[i], 2) for i = 1:NB)
   @assert size(∂A, 2) == length(basis)
   @assert length(BB) == NB 
   @assert length(∂BB) == NB 

   # ∂A_loc = zeros(eltype(∂A), nT)

   @inbounds for (iA, ϕ) in enumerate(basis.spec)
      # @simd ivdep for t = 1:nT 
      #    ∂A_loc[t] = ∂A[t, iA]
      # end
      @simd ivdep for j = 1:nX 
         ∂A_iA = ∂A[target[j], iA] # ∂A_loc[target[j] ] 
         b = ntuple(Val(NB)) do i 
            @inbounds BB[i][j, ϕ[i]] 
         end 
         g = _prod_grad(b, Val(NB))
         for i = 1:NB 
            ∂BB[i][j, ϕ[i]] = muladd(∂A_iA, g[i], ∂BB[i][j, ϕ[i]])
         end
      end 
   end
   return nothing 
end


# --------------------- connect with ChainRules 
# todo ... 

function rrule(::typeof(evalpool), basis::PooledSparseProduct{NB}, BB::Tuple) where {NB}
   A = evalpool(basis, BB)

   function pb(Δ)
      ∂BB = _pullback_evalpool(Δ, basis, BB)
      return NoTangent(), NoTangent(), ∂BB
   end 

   return A, pb 
end

# --------------------- connect with Lux 


struct PooledSparseProductLayer{NB} <: AbstractExplicitLayer 
   basis::PooledSparseProduct{NB}
end

lux(basis::PooledSparseProduct) = PooledSparseProductLayer(basis)

initialparameters(rng::AbstractRNG, layer::PooledSparseProductLayer) = 
      NamedTuple() 

initialstates(rng::AbstractRNG, layer::PooledSparseProductLayer) = 
      NamedTuple()

(l::PooledSparseProductLayer)(BB, ps, st) = 
      evalpool(l.basis, BB), st 
