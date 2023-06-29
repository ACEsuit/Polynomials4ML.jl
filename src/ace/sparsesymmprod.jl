
using LoopVectorization

using ChainRulesCore
using ChainRulesCore: NoTangent

struct SparseSymmProd <: AbstractPoly4MLBasis
   dag::SparseSymmProdDAG
   proj::Vector{Int}
   @reqfields
end

function SparseSymmProd(spec::AbstractVector{<: Union{Tuple, AbstractVector}}; kwargs...)
   dag = SparseSymmProdDAG(spec; kwargs...)
   return SparseSymmProd(dag, dag.projection, _make_reqfields()... )
end

Base.length(basis::SparseSymmProd) = length(basis.proj)

reconstruct_spec(basis::SparseSymmProd) = reconstruct_spec(basis.dag)[basis.proj]

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


# -------------- kernels  (these are really just interfaces as well...)

# this one does both batched and unbatched
function evaluate!(AA, basis::SparseSymmProd, A)
   AAdag = evaluate(basis.dag, A)
   _project!(AA, basis.proj, AAdag)
   release!(AAdag)
   return AA 
end

# serial projection 
function _project!(BB, proj::Vector{<: Integer}, AA::AbstractVector{<: Number})
   @inbounds for i = 1:length(proj)
      BB[i] = AA[proj[i]]
   end
   return nothing
end

# batched projection 
function _project!(BB, proj::Vector{<: Integer}, AA::AbstractMatrix{<: Number})
   nX = size(AA, 1)
   @assert size(BB, 1) >= nX
   @inbounds for i = 1:length(proj)
      p_i = proj[i]
      @simd ivdep for j = 1:nX
         BB[j, i] = AA[j, p_i]
      end
   end
   return nothing
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

# try non-allocating lux
# (l::PolyLuxLayer{SparseSymmProd})(A, ps, st) = LuxCore.apply(l, A, ps, st)

function evaluate(l::PolyLuxLayer{SparseSymmProd}, A::AbstractVector{T}, ps, st) where {T}
   AA = acquire!(l.basis.tmp, :AA, (length(l),), T)
   evaluate!(AA, l.basis, A)
   return AA, st
end

function evaluate(l::PolyLuxLayer{SparseSymmProd}, A::AbstractMatrix{T}, ps, st) where {T}
   nX = size(A, 1)
   AA = acquire!(l.basis.tmp, :AAbatch, (nX, length(l)), T)
   evaluate!(AA, l.basis, A)
   return AA, st
end

function ChainRulesCore.rrule(::typeof(LuxCore.apply), l::PolyLuxLayer{SparseSymmProd}, A, ps, st)
   AAdag = evaluate(l.basis.dag, A)
   AA = AAdag[:, l.basis.proj]
   return (AA, st), Δ -> (NoTangent(), NoTangent(), _pullback(Δ[1], l.basis, A, AA, AAdag), NoTangent(), NoTangent())
end

