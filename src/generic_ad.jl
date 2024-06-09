# ---------------------------------------------------------------
# general rrules and frules interface for AbstractP4MLBasis 


function whatalloc(::typeof(pullback_evaluate!), 
                    ∂P, basis::AbstractP4MLBasis, X::AbstractVector)
   T∂X = promote_type(_gradtype(basis, X), eltype(∂P))
   return (T∂X, length(X))
end

function pullback_evaluate!(∂X, 
                  ∂P, basis::AbstractP4MLBasis, X::AbstractVector; 
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

function rrule(::typeof(evaluate), 
                  basis::AbstractP4MLBasis, 
                  X::AbstractVector)
   P = evaluate(basis, X)
   # TODO: here we could do evaluate_ed, but need to think about how this 
   #       works with the kwarg trick above...
   return P, ∂P -> (NoTangent(), NoTangent(), pullback_evaluate(∂P, basis, X))
end


#= 
function whatalloc(::typeof(pb_pb_evaluate!), 
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


function pb_pb_evaluate!(∂²P, ∂²X,   # output 
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


function rrule(::typeof(pullback_evaluate),
   ∂P, basis::AbstractP4MLBasis, X::AbstractVector{<: Real})
∂X = pullback_evaluate(∂P, basis, X)
function _pb(∂2)
∂∂P, ∂X = pb_pb_evaluate(∂2, ∂P, basis, X)
return NoTangent(), ∂∂P, NoTangent(), ∂X             
end
return ∂X, _pb 
end
=#


# -------------------------------------------------------------
# general rrules and frules for AbstractP4MLTensor 


function rrule(::typeof(evaluate), 
                  basis::AbstractP4MLTensor, 
                  X)
   P = evaluate(basis, X)
   return P, ∂P -> (NoTangent(), NoTangent(), pullback_evaluate(∂P, basis, X))
end

