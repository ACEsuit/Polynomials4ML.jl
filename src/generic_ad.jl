# ---------------------------------------------------------------
# general rrules and frules interface for AbstractP4MLBasis 

_promote_grad_type(::Type{T}, ::Type{S}
                  ) where {T <: Number, S <: Number} = 
         promote_type(T, S)

_promote_grad_type(::Type{SVector{D, T}}, ::Type{S}
                  ) where {D, T, S <: Number} = 
         SVector{D, promote_type(T, S)}

function whatalloc(::typeof(pullback!), 
                    ∂P, basis::AbstractP4MLBasis, X::BATCH)
   T∂X = _promote_grad_type(_gradtype(basis, X), eltype(∂P))
   return (T∂X, length(X))
end

function pullback!(∂X, 
                   ∂P, basis::AbstractP4MLBasis, X::BATCH; 
                   dP = evaluate_ed(basis, X)[2] )
   @assert size(∂P) == size(dP) == (length(X), length(basis))
   @assert length(∂X) == length(X)
   # manual loops to avoid any broadcasting of StrideArrays 
   # ∂_xa ( ∂P : P ) = ∑_ij ∂_xa ( ∂P_ij * P_ij ) 
   #                 = ∑_ij ∂P_ij * ∂_xa ( P_ij )
   #                 = ∑_ij ∂P_ij * dP_ij δ_ia
   for n = 1:size(dP, 2)
      @simd ivdep for a = 1:length(X)
            ∂X[a] += dP[a, n] * ∂P[a, n]
      end
   end
   return ∂X
end

pullback(∂X, l::AbstractP4MLBasis, args...) = 
      _with_safe_alloc(pullback!, ∂X, l, args...)


function rrule(::typeof(evaluate), 
                  basis::AbstractP4MLBasis, 
                  X::BATCH)
   P = evaluate(basis, X)
   # TODO: here we could do evaluate_ed, but need to think about how this 
   #       works with the kwarg trick above...
   return P, ∂P -> (NoTangent(), NoTangent(), pullback(∂P, basis, X))
end

# -------- pullback w.r.t. params 

EMPTY_NT = typeof(NamedTuple())

pullback_ps(∂P, basis::AbstractP4MLBasis, X::BATCH, ps::Nothing, st) = 
         NoTangent() 

pullback_ps(∂P, basis::AbstractP4MLBasis, X::BATCH, ps::EMPTY_NT, st) = 
         NamedTuple() 


function rrule(::typeof(evaluate), 
               basis::AbstractP4MLBasis, 
               X::BATCH, 
               ps, st)
   P, dP = evaluate_ed(basis, X, ps, st)

   function _pb(_∂P)
      ∂P = unthunk(_∂P)
      # compute the pullback w.r.t. X 
      T∂X, N∂X = whatalloc(pullback!, ∂P, basis, X)
      ∂X = zeros(T∂X, N∂X)
      pullback!(∂X, ∂P, basis, X; dP = dP)
      
      ∂X = pullback(∂P, basis, X)
      ∂ps = pullback_ps(∂P, basis, X, ps, st)

      return NoTangent(), NoTangent(), ∂X, ∂ps, NoTangent() 
   end

   return P, _pb 
end


# function rrule(::typeof(evaluate_ed), 
#                basis::AbstractP4MLBasis, 
#                X::BATCH, 
#                ps, st)
#    P, dP = evaluate_ed(basis, X, ps, st)

#    return (P, dP), ∂P -> (NoTangent(), NoTangent(), pullback(∂P, basis, X))
# end


#= 
function whatalloc(::typeof(pullback2!), 
                   ∂∂X, ∂P, basis::AbstractP4MLBasis, X::AbstractVector)
   Nbasis = length(basis)
   Nx = length(X)                        
   @assert ∂∂X isa AbstractVector 
   @assert length(∂∂X) == Nx
   @assert size(∂P) == (Nx, Nbasis)
   T∂²P = promote_type(_valtype(basis, X), eltype(∂P), eltype(∂∂X))
   T∂²X = promote_type(_gradtype(basis, X), eltype(∂P), eltype(∂∂X))
   return (T∂²P, Nx, Nbasis), (T∂²X, Nx)
end


function pullback2!(∂²P, ∂²X,   # output 
                         ∂∂X,        # input / perturbation of ∂X
                         ∂P, basis::AbstractP4MLBasis,   # inputs 
                         X::AbstractVector{<: Real})
   @no_escape begin                          
      P, dP, ddP = @withalloc evaluate_ed2!(basis, X)

      for n = 1:Nbasis 
         @simd ivdep for a = 1:Nx 
            ∂²P[a, n] = ∂∂X[a] * dP[a, n]
            ∂²X[a] += ∂∂X[a] * ddP[a, n] * ∂P[a, n]
         end
      end
   end

   return ∂²P, ∂²X
end


function rrule(::typeof(pullback),
   ∂P, basis::AbstractP4MLBasis, X::AbstractVector{<: Real})
∂X = pullback(∂P, basis, X)
function _pb(∂2)
∂∂P, ∂X = pullback2(∂2, ∂P, basis, X)
return NoTangent(), ∂∂P, NoTangent(), ∂X             
end
return ∂X, _pb 
end
=#
