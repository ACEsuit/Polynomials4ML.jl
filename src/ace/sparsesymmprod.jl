
using LoopVectorization

using ChainRulesCore
using ChainRulesCore: NoTangent

export SparseSymmProd

@doc raw"""
`SparseSymmProd` : sparse symmetric product with entries stored as tuples. 
Input is a vector `A`; each entry of the output vector `AA` is of the form 
```math
 {\bm A}_{i_1, \dots, i_N} = \prod_{t = 1}^N A_{i_t}.
```

### Constructor 
```julia 
SparseSymmProd(spec)
```
where `spec` is a list of tuples or vectors, each of which specifies an `AA`
basis function as described above. For example, 
```julia 
spec = [ (1,), (2,), (1,1), (1,2), (2,2), 
         (1,1,1), (1,1,2), (1,2,2), (2,2,2) ]
basis = SparseSymmProd(spec)         
```
defines a basis of 9 functions, 
```math 
[ A_1, A_2, A_1^2, A_1 A_2, A_2^2, A_1^3, A_1^2 A_2, A_1 A_2^2, A_2^3 ]
```
"""
struct SparseSymmProd{ORD, TS} <: AbstractPoly4MLBasis
   specs::TS
   ranges::NTuple{ORD, UnitRange{Int}}
   hasconst::Bool 
   # --------------
   @reqfields
end


function SparseSymmProd(spec::AbstractVector{<: Union{Tuple, AbstractVector}}; kwargs...)
   if !issorted(spec, by=length) 
      spec = sort(spec, by=length)
   end
   if length(spec[1]) == 0
      hasconst = true 
      spec = spec[2:end]
   else
      hasconst = false
   end

   MAXORD = length(spec[end])
   specs = ntuple(N -> Vector{NTuple{N, Int}}(), MAXORD)
   for b in spec 
      N = length(b) 
      push!(specs[N], tuple(sort([b...])...))
   end
   ranges = [] 
   idx = Int(hasconst)
   for N = 1:MAXORD
      len = length(specs[N])
      push!(ranges, (idx+1):(idx+len))
      idx += len
   end
   return SparseSymmProd(specs, tuple(ranges...), hasconst, _make_reqfields()...)   
end

Base.length(basis::SparseSymmProd) = sum(length, basis.specs) + basis.hasconst

function reconstruct_spec(basis::SparseSymmProd) 
   spec = [ [ bb... ] for bb in vcat(basis.specs...) ]
   if basis.hasconst
      prepend!(spec, [Int[],])
   end
   return spec 
end
# -------------- evaluation interfaces 

_valtype(basis::SparseSymmProd, ::Type{T}) where {T} = T

(basis::SparseSymmProd)(args...) = evaluate(basis, args...)

function evaluate(basis::SparseSymmProd, A::AbstractVector{T}) where {T}
   AA = acquire!(basis.pool, :AA, (length(basis),), T)
   evaluate!(AA, basis, A)
   return AA
end

function evaluate(basis::SparseSymmProd, A::AbstractMatrix{T}) where {T}
   nX = size(A, 1)
   AA = acquire!(basis.pool, :AAbatch, (nX, length(basis)), T)
   evaluate!(AA, basis, A)
   return AA
end


# -------------- kernels for simple evaluation 

using Base.Cartesian: @nexprs 
using ObjectPools: FlexCachedArray

__view_AA(AA::FlexCachedArray, basis, N) = __view_AA(unwrap(AA), basis, N)
__view_AA(AA::AbstractVector, basis, N) = (@view AA[basis.ranges[N]])
__view_AA(AA::AbstractMatrix, basis, N) = (@view AA[:, basis.ranges[N]])

# this one does both batched and unbatched
@generated function evaluate!(AA, basis::SparseSymmProd{ORD}, A) where {ORD} 
   quote 
      if basis.hasconst; _evaluate_AA_const!(AA); end
      @nexprs $ORD i -> _evaluate_AA!( __view_AA(AA, basis, i), 
                                       basis.specs[i], 
                                       A)
      return AA
   end
end

function _evaluate_AA_const!(AA::AbstractVector) 
   AA[1] = one(eltype(AA))
   return nothing
end

function _evaluate_AA_const!(AA::AbstractMatrix) 
   AA[:, 1] .= one(eltype(AA))
   return nothing
end

function _evaluate_AA!(AA, spec::Vector{NTuple{N, Int}}, A::AbstractVector) where {N}
   @assert length(AA) >= length(spec)
   @inbounds for (i, ϕ) in enumerate(spec)
      aa = ntuple(i -> A[ϕ[i]], N)
      AA[i] = prod(aa)
   end
   return nothing 
end

function _evaluate_AA!(AA, spec::Vector{NTuple{N, Int}}, A::AbstractMatrix) where {N}
   nX = size(A, 1)
   @assert size(AA, 1) >= nX 
   @assert size(AA, 2) >= length(spec)
   @inbounds for (i, ϕ) in enumerate(spec)
      @simd ivdep for j = 1:nX
         aa = ntuple(i -> A[j, ϕ[i]], N)
         AA[j, i] = prod(aa)
      end
   end
   return nothing 
end


# ------- 

import ChainRulesCore: rrule, NoTangent 

function rrule(::typeof(evaluate), basis::SparseSymmProd, A)
   AA = evaluate(basis, A)
   return AA, Δ -> (NoTangent(), NoTangent(), _pb_evaluate(basis, Δ, A))
end


@generated function _pb_evaluate(
                        basis::SparseSymmProd{ORD}, 
                        Δ,  # differential
                        A   # input 
                        ) where {ORD}
   quote                         
      TG = promote_type(eltype(Δ), eltype(A))
      gA = zeros(TG, size(A))
      @nexprs $ORD N -> _pb_evaluate_pbAA!(
                              gA, 
                              __view_AA(Δ, basis, N), 
                              basis.specs[N], 
                              A)
      return gA
   end 
end

# _pb_evaluate_pbAA_const!(gA::AbstractVector) = (gA[1] .= 0; nothing)
# _pb_evaluate_pbAA_const!(gA::AbstractMatrix) = (gA[:, 1] .= 0; nothing)

function _pb_evaluate_pbAA!(gA::AbstractVector, ΔN::AbstractVector, 
                            spec::Vector{NTuple{N, Int}}, 
                            A) where {N}
   # we compute ∇_A w.r.t. the expression ∑_i Δ[i] * AA[i]                             
   for (i, ϕ) in enumerate(spec)
      aa = ntuple(i -> A[ϕ[i]], N)
      pi, gi = _static_prod_ed(aa) 
      for j = 1:N 
         gA[ϕ[j]] += ΔN[i] * gi[j]
      end
   end
   return nothing 
end 

function _pb_evaluate_pbAA!(gA, ΔN::AbstractMatrix, 
                            spec::Vector{NTuple{N, Int}}, 
                            A::AbstractMatrix) where {N}
   nX = size(A, 1)                            
   for (i, ϕ) in enumerate(spec)
      for j = 1:nX 
         aa = ntuple(i -> A[j, ϕ[i]], N)
         pi, gi = _static_prod_ed(aa) 
         for t = 1:N 
            gA[j, ϕ[t]] += ΔN[j, i] * gi[t]
         end
      end
   end
   return nothing 
end


function rrule(::typeof(_pb_evaluate), basis::SparseSymmProd, ΔAA, A)
   uA = _pb_evaluate(basis, ΔAA, A)
   return uA, Δ² -> (NoTangent(), NoTangent(), 
                     _pb_pb_evaluate(basis, Δ², ΔAA, A)...)
end


@generated function _pb_pb_evaluate(basis::SparseSymmProd{ORD}, Δ², ΔAA, A)  where {ORD}
   quote 
      TG = promote_type(eltype(Δ²), eltype(ΔAA), eltype(A))
      gΔAA = zeros(TG, size(ΔAA))
      gA = zeros(TG, size(A))
      @nexprs $ORD N -> _pb_pb_evaluate_AA!(basis.specs[N], 
                              __view_AA(gΔAA, basis, N), gA,   # outputs (gradients)
                              Δ²,                              # differential 
                              __view_AA(ΔAA,  basis, N), A,    # inputs 
                              )
      return gΔAA, gA
   end 
end 

function _pb_pb_evaluate_AA!(spec::Vector{NTuple{N, Int}}, 
                             gΔAA, gA, 
                             Δ², 
                             ΔN::AbstractVector, A::AbstractVector) where {N}
   # We wish to compute ∇_Δ and ∇_A w.r.t.  the expression 
   #         ∑ₖ Δ²ₖ * ∇_{Aₖ} (Δ ⋅ AA)    (Δ = ΔN)
   #      =  ∑ᵢ Δᵢ * ∇̃ AA[i] 
   # where   ∇̃ = ∑_k Δ²ₖ * ∇_Aₖ
   #   here k = 1,...,#A and i = 1,...,#AA 

   @assert size(gA) == size(A) 
   @assert length(gΔAA) >= length(spec)
   @assert length(ΔN) >= length(spec)
   @assert length(Δ²) >= length(A)

   @inbounds for (i, ϕ) in enumerate(spec)
      A_ϕ = ntuple(t -> A[ϕ[t]], N)
      Δ²_ϕ = ntuple(t -> Δ²[ϕ[t]], N)
      p_i, g_i, u_i = _pb_grad_static_prod(Δ²_ϕ, A_ϕ)
      gΔAA[i] = sum(g_i .* Δ²_ϕ) 
      for t = 1:N 
         gA[ϕ[t]] += u_i[t] * ΔN[i]
      end
   end
   return nothing 
end


function _pb_pb_evaluate_AA!(spec::Vector{NTuple{N, Int}}, 
                             gΔAA, gA, 
                             Δ², 
                             ΔN::AbstractMatrix, A::AbstractMatrix) where {N}
   nX = size(A, 1)
   for (i, ϕ) in enumerate(spec)
      for j = 1:nX
         A_ϕ = ntuple(t -> A[j, ϕ[t]], N)
         Δ²_ϕ = ntuple(t -> Δ²[j, ϕ[t]], N)
         p_i, g_i, u_i = _pb_grad_static_prod(Δ²_ϕ, A_ϕ)
         gΔAA[j, i] = sum(g_i .* Δ²_ϕ)
         for t = 1:N 
            gA[j, ϕ[t]] += u_i[t] * ΔN[j, i]
         end
      end
   end
   return nothing 
end


# -------------- Lux integration 
# it needs an extra lux interface reason as in the case of the `basis` 
# should it not be enough to just overload valtype? 

function evaluate(l::PolyLuxLayer{<: SparseSymmProd}, A::AbstractVector{T}, ps, st) where {T}
   AA = acquire!(st.pool, :AA, (length(l),), T)
   evaluate!(AA, l.basis, A)
   return AA, st
end

function evaluate(l::PolyLuxLayer{<: SparseSymmProd}, A::AbstractMatrix{T}, ps, st) where {T}
   nX = size(A, 1)
   AA = acquire!(st.pool, :AAbatch, (nX, length(l)), T)
   evaluate!(AA, l.basis, A)
   return AA, st
end
