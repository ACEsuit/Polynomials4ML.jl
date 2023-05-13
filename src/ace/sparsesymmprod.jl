
using LoopVectorization

struct SparseSymmProd{T} 
   dag::SparseSymmProdDAG{T} 
   proj::Vector{Int}
   pool::ArrayPool{FlexArrayCache}
end

function SparseSymmProd(spec::AbstractVector{<: AbstractVector}; T = Float64, kwargs...)
   dag = SparseSymmProdDAG(spec; T=T, kwargs...)
   return SparseSymmProd(dag, dag.projection, ArrayPool(FlexArrayCache))
end

Base.length(basis::SparseSymmProd) = length(basis.proj)

(basis::SparseSymmProd)(args...) = evaluate(basis, args...)

reconstruct_spec(basis::SparseSymmProd) = reconstruct_spec(basis.dag)[basis.proj]

# -------------- evaluation interfaces 

function evaluate(basis::SparseSymmProd, A::AbstractVector{T}) where {T}
   AA = acquire!(basis.pool, :AA, (length(basis),), T)
   evaluate!(parent(AA), basis, A)
   return AA
end

function evaluate(basis::SparseSymmProd, A::AbstractMatrix{T}) where {T}
   nX = size(A, 1)
   AA = acquire!(basis.pool, :AAbatch, (nX, length(basis)), T)
   evaluate!(parent(AA), basis, A)
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
function _project!(BB, proj::Vector{<: Integer}, AA::AbstractVector)
   @inbounds for i = 1:length(proj)
      BB[i] = AA[proj[i]]
   end
   return nothing
end

# batched projection 
function _project!(BB, proj::Vector{<: Integer}, AA::AbstractMatrix)
   nX = size(AA, 1)
   @assert size(BB, 1) >= nX
   @inbounds for i = 1:length(proj)
      p_i = proj[i]
      for j = 1:nX
         BB[j, i] = AA[j, p_i]
      end
   end
   return nothing
end


# -------------- Chainrules integration 

function _pullback(Δ, basis::SparseSymmProd, A, AA, AAdag)
   Δdag = zeros(eltype(Δ), length(AAdag))
   Δdag[basis.proj] .= Δ
   T = promote_type(eltype(Δdag), eltype(AAdag))
   ΔA = zeros(T, length(A))
   pullback_arg!(ΔA, Δdag, basis.dag, AAdag)
   return ΔA
end

function rrule(::typeof(evaluate), basis::SparseSymmProd, A::AbstractVector)
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

# -------------- Lux integration 

struct SparseSymmProdLayer{T} <: AbstractExplicitLayer
   basis::SparseSymmProd{T}
end

function lux(basis::SparseSymmProd) 
   return SparseSymmProdLayer(basis)
end

Base.length(l::SparseSymmProdLayer) = length(l.basis)

initialparameters(rng::AbstractRNG, l::SparseSymmProdLayer) = NamedTuple() 

initialstates(rng::AbstractRNG, l::SparseSymmProdLayer) = NamedTuple()

(l::SparseSymmProdLayer)(A, ps, st) = evaluate(l.basis, A), st 





