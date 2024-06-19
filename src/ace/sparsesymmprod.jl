
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

Base.show(io::IO, basis::SparseSymmProd{ORD}) where {ORD} = 
      print(io, "SparseSymmProd(order=$(ORD), length = $(length(basis)))")

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


# ----------------------------------- 
#  pullback 

import ChainRulesCore: rrule, NoTangent 

function whatalloc(::typeof(pullback!), 
                   ∂AA, basis::SparseSymmProd, A)
   T∂A = promote_type(eltype(∂AA), eltype(A))
   return (T∂A, size(A)... )
end


@generated function pullback!(∂A,  
                                 ∂AA, basis::SparseSymmProd{ORD}, A
                                 ) where {ORD}
   quote
      fill!(∂A, zero(eltype(∂A)))
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


# ----------------------------------- 
#   reverse-over-reverse 
#
#  AA = evaluate(basis, A) 
#  ∇_A = pullback(∂AA, basis, A)
#  ∇_∂AA, ∇_A = pullback2(∂∇A, ∂AA, basis, A)
#

function whatalloc(::typeof(pullback2!), 
                   ∂∇A,   # cotangent to be pulled back 
                   ∂AA,   # cotangent from pullback(∂AA, basis, A)
                   basis::SparseSymmProd, A)
   T = promote_type(eltype(∂∇A), eltype(∂AA), eltype(A))
   return (T, size(∂AA)...), (T, size(A)...)
end



function pullback2!(∇_∂AA, ∇_A, 
                    ∂∇A, ∂AA, basis, A)
   @assert size(∂∇A) == size(A)                     
   T = promote_type(eltype(∂∇A), eltype(∂AA), eltype(A))
   d = Dual{T}(zero(T), one(T))
   DT = typeof(d)
   @no_escape begin 
      A_d = @alloc(DT, size(A)...)
      @inbounds for i = 1:length(A) 
         A_d[i] = A[i] + d * ∂∇A[i]
      end
      AA_d = @withalloc evaluate!(basis, A_d)
      # ∇A_d = pullback(∂AA, basis, A_d)
      ∇A_d = @withalloc pullback!(∂AA, basis, A_d)
      @inbounds for i = 1:length(AA_d)
         ∇_∂AA[i] = extract_derivative(eltype(∇_∂AA), AA_d[i])
      end
      @inbounds for i = 1:length(∇A_d)
         ∇_A[i] = extract_derivative(eltype(∇_A), ∇A_d[i])
      end
   end
   return ∇_∂AA, ∇_A
end



# -------------- Pushforwards / frules  

using ForwardDiff

function whatalloc(::typeof(pushforward!), 
                   basis::SparseSymmProd, 
                   A::AbstractVector, ΔA::AbstractVector)
   nAA = length(basis)
   TAA = eltype(A)
   T∂AA = promote_type(TAA, eltype(ΔA))
   return (TAA, nAA), (T∂AA, nAA)
end

function whatalloc(::typeof(pushforward!), 
                   basis::SparseSymmProd, 
                   A::AbstractMatrix, ΔA::AbstractMatrix)
   nAA = length(basis)
   TAA = eltype(A)
   T∂AA = promote_type(TAA, eltype(ΔA))
   nX = size(A, 1)
   return (TAA, nX, nAA), (T∂AA, nX, nAA)
end


function pushforward!(AA, ∂AA, basis::SparseSymmProd, A, ∂A)
   @assert size(∂A) == size(A)
   @assert size(∂AA) == size(AA)
   T = promote_type(eltype(A), eltype(∂A))
   d = Dual{T}(zero(T), one(T))
   DT = typeof(d)
   @no_escape begin 
      A_d = @alloc(DT, size(A)...)
      @inbounds for i = 1:length(A) 
         A_d[i] = A[i] + d * ∂A[i]
      end
      AA_d = @withalloc evaluate!(basis, A_d)
      @assert length(AA_d) <= length(AA) && length(AA_d) == length(∂AA)
      @inbounds for i = 1:length(AA_d)
         AA[i] = ForwardDiff.value(AA_d[i])
         ∂AA[i] = ForwardDiff.extract_derivative(eltype(∂AA), AA_d[i])
      end 
   end
   return AA, ∂AA
end


# ------------------------------------------
#  ChainRules integration 

function rrule(::typeof(pullback), ∂AA, basis::SparseSymmProd, A) 
   ∂A = pullback(∂AA, basis, A)
   function pb(∂∂A)
      ∂²∂AA, ∂²A = pullback2(∂∂A, ∂AA, basis, A)
      return NoTangent(), ∂²∂AA, NoTangent(), ∂²A
   end
   return ∂A, pb
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
