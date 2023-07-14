
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
   @inbounds for (i, ϕ) in enumerate(spec)
      aa = ntuple(i -> A[ϕ[i]], N)
      AA[i] = prod(aa)
   end
   return AA
end





# -------------- Chainrules integration 

function _pullback(Δ, basis::SparseSymmProd, A::AbstractVector, AA, AAdag)
   Δdag = zeros(eltype(Δ), length(AAdag))
   Δdag[basis.proj] .= Δ
   T = promote_type(eltype(Δdag), eltype(AAdag))
   ΔA = zeros(T, length(A))
   pullback_arg!(ΔA, Δdag, basis.dag, AAdag)
   return ΔA
end


function _pullback(Δ, basis::SparseSymmProd, A::AbstractMatrix, AA, AAdag)
   Δdag = zeros(eltype(Δ), size(AAdag)...)
   Δdag[:, basis.proj] .= Δ
   T = promote_type(eltype(Δdag), eltype(AAdag))
   ΔA = zeros(T, size(A)...)
   pullback_arg!(ΔA, Δdag, basis.dag, AAdag)
   return ΔA
end


function ChainRulesCore.rrule(::typeof(evaluate), basis::SparseSymmProd, A::AbstractVector)
   AAdag = evaluate(basis.dag, A)
   AA = AAdag[basis.proj]

   # function evaluate_pullback(Δ)
   #    Δdag = zeros(eltype(Δ), length(AAdag))
   #    Δdag[basis.proj] .= Δ
   #    T = promote_type(eltype(Δdag), eltype(AAdag))
   #    ΔA = zeros(T, length(A))
   #    pullback_arg!(ΔA, Δdag, basis.dag, AAdag)
   #    return NoTangent(), NoTangent(), ΔA
   # end
   return AA, Δ -> (NoTangent(), NoTangent(), _pullback(Δ, basis, A, AA, AAdag))
end

function ChainRulesCore.rrule(::typeof(evaluate), basis::SparseSymmProd, A::AbstractMatrix)
   AAdag = evaluate(basis.dag, A)
   AA = AAdag[:, basis.proj]
   return AA, Δ -> (NoTangent(), NoTangent(), _pullback(Δ, basis, A, AA, AAdag))
end

# -------------- Lux integration 

# it needs an extra lux interface reason as in the case of the `basis` 
function evaluate(l::PolyLuxLayer{SparseSymmProd}, A::AbstractVector{T}, ps, st) where {T}
   AA = acquire!(st.pool, :AA, (length(l),), T)
   evaluate!(AA, l.basis, A)
   return AA, st
end

function evaluate(l::PolyLuxLayer{SparseSymmProd}, A::AbstractMatrix{T}, ps, st) where {T}
   nX = size(A, 1)
   AA = acquire!(st.pool, :AAbatch, (nX, length(l)), T)
   evaluate!(AA, l.basis, A)
   return AA, st
end
