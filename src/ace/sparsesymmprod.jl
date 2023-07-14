
using LoopVectorization

using ChainRulesCore
using ChainRulesCore: NoTangent

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
      push!(specs[N], sort(tuple(b...)))
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


# -------------- kernels  

using Base.Cartesian: @nexprs 

# this one does both batched and unbatched
@generated function evaluate!(_AA, basis::SparseSymmProd{ORD}, A) where {ORD} 
   quote 
      AA = parent(_AA)
      if basis.hasconst; AA[1] = one(eltype(AA)); end
      @nexprs $ORD i -> _evaluate_AA!( (@view AA[basis.ranges[i]]), 
                                       basis.specs[i], 
                                       A)
      return _AA 
   end
end

function _evaluate_AA!(AA, spec::Vector{NTuple{N, Int}}, A) where {N}
   for (i, ϕ) in enumerate(spec)
      aa = ntuple(i -> A[ϕ[i]], N)
      AA[i] = prod(aa)
   end
   return nothing 
end


# -------------- Chainrules integration 

import ChainRulesCore: rrule, NoTangent 

function rrule(::typeof(evaluate), basis::SparseSymmProd, A::AbstractVector)
   AA = evaluate(basis, A)
   return AA, Δ -> (NoTangent(), NoTangent(), _pb_evaluate(basis, Δ, A))
end


@generated function _pb_evaluate(
                        basis::SparseSymmProd{ORD}, 
                        Δ::AbstractVector{TΔ},  # differential
                        _A::AbstractVector{TA}   # input 
                        ) where {ORD, TΔ, TA}
   TG = promote_type(TΔ, TA) 
   quote                         
      @assert length(Δ) == length(basis)
      A = parent(_A)
      gA = zeros($TG, length(A))

      if basis.hasconst
         gA[1] = 0 
      end

      @nexprs $ORD i -> _pb_evaluate_pbAA!(gA, (@view Δ[basis.ranges[i]]), basis.specs[i], A)

      return gA 
   end 
end

function _pb_evaluate_pbAA!(gA::AbstractVector, ΔN::AbstractVector, spec::Vector{NTuple{N, Int}}, A) where {N}
   for (i, ϕ) in enumerate(spec)
      aa = ntuple(i -> A[ϕ[i]], N)
      pi, gi = _grad_static_prod(aa) 
      for j = 1:N 
         gA[ϕ[j]] += ΔN[i] * gi[j]
      end
   end
   return nothing 
end 




