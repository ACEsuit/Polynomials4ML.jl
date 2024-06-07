
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
struct SparseSymmProd{ORD, TS} <: AbstractP4MLTensor
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

function whatalloc(::typeof(evaluate!), 
                   basis::SparseSymmProd, A::AbstractVector{T}) where {T}
   VT = _valtype(basis, T)
   return (VT, length(basis))
end

function whatalloc(::typeof(evaluate!), 
                   basis::SparseSymmProd, A::AbstractMatrix{T}) where {T}
   VT = _valtype(basis, T)
   return (VT, size(A, 1), length(basis))
end

# -------------- kernels for simple evaluation 

using Base.Cartesian: @nexprs 

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
         aa = ntuple(i -> @inbounds(A[j, ϕ[i]]), N)
         AA[j, i] = prod(aa)
      end
   end
   return nothing 
end


# ------- 

import ChainRulesCore: rrule, NoTangent 

function whatalloc(::typeof(pullback_evaluate!), 
                   ∂AA, basis::SparseSymmProd, A)
   T∂A = promote_type(eltype(∂AA), eltype(A))
   return (T∂A, size(A)... )
end

# TODO: REMOVE
# function pullback_evaluate(∂AA, basis::SparseSymmProd, A)
#    ∂A = zeros(T∂A, size(A))
#    return pullback_evaluate!(∂A, ∂AA, basis, A)
# end

@generated function pullback_evaluate!(∂A,  
                                 ∂AA, basis::SparseSymmProd{ORD}, A
                                 ) where {ORD}
   quote
      @nexprs $ORD N -> _pb_evaluate_pbAA!(
                              ∂A, 
                              __view_AA(∂AA, basis, N), 
                              basis.specs[N], 
                              A)
      return ∂A
   end 
end


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
         aa = ntuple(i -> @inbounds(A[j, ϕ[i]]), N)
         pi, gi = _static_prod_ed(aa) 
         for t = 1:N 
            gA[j, ϕ[t]] += ΔN[j, i] * gi[t]
         end
      end
   end
   return nothing 
end


#=
@generated function pb_pb_evaluate(Δ², ΔAA, basis::SparseSymmProd{ORD}, A)  where {ORD}
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
=#


# -------------- Pushforwards / frules  

# TODO: REMOVE
# function pushforward_evaluate(basis::SparseSymmProd, 
#                               A::AbstractVector{<: Number}, 
#                               ΔA::AbstractMatrix)
#    nAA = length(basis)                       
#    TAA = eltype(A)
#    AA = zeros(TAA, nAA)
#    T∂AA = _my_promote_type(TAA, eltype(ΔA))
#    ∂AA = zeros(T∂AA, nAA, size(ΔA, 2))
#    pushforward_evaluate!(AA, ∂AA, basis, A, ΔA)
#    return AA, ∂AA
# end

function whatalloc(::typeof(pushforward_evaluate!), 
                   basis::SparseSymmProd, A, ΔA)
   nAA = length(basis)                       
   TAA = eltype(A)
   T∂AA = _my_promote_type(TAA, eltype(ΔA))
   return (TAA, nAA), (T∂AA, nAA, size(ΔA, 2))
end

@generated function pushforward_evaluate!(AA, ∂AA, basis::SparseSymmProd{NB}, A, ΔA) where {NB}
   quote 
      if basis.hasconst; error("no implementation with hasconst"); end 
      Base.Cartesian.@nexprs $NB N -> _pfwd_AA_N!(AA, ∂AA, A, ΔA, basis.ranges[N], basis.specs[N])
      return AA, ∂AA
   end
end

function _pfwd_AA_N!(AA, ∂AA, A, ΔA, 
                     rg_N, spec_N::Vector{NTuple{N, Int}}) where {N}
   nX = size(ΔA, 2)                     
   for (i, bb) in zip(rg_N, spec_N)
      aa = ntuple(t -> A[bb[t]], N)
      ∏aa, ∇∏aa = Polynomials4ML._static_prod_ed(aa)
      AA[i] = ∏aa
      for t = 1:N, j = 1:nX
         ∂AA[i, j] += ∇∏aa[t] * ΔA[bb[t], j]
      end
   end
end 



# -------------- Lux integration 
# it needs an extra lux interface reason as in the case of the `basis` 
# should it not be enough to just overload valtype? 

# function evaluate(l::PolyLuxLayer{<: SparseSymmProd}, A::AbstractVector{T}, ps, st) where {T}
#    AA = acquire!(st.pool, :AA, (length(l),), T)
#    evaluate!(AA, l.basis, A)
#    return AA, st
# end

# function evaluate(l::PolyLuxLayer{<: SparseSymmProd}, A::AbstractMatrix{T}, ps, st) where {T}
#    nX = size(A, 1)
#    AA = acquire!(st.pool, :AAbatch, (nX, length(l)), T)
#    evaluate!(AA, l.basis, A)
#    return AA, st
# end
