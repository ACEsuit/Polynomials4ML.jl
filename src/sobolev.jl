using LinearAlgebra: eigen, Symmetric, mul!

# TODO: this is a temporary construction. It might be better to write 
#       a simple chain structure that is more flexible but implements 
#       the P4ML interface. 

struct TransformedBasis{TB, TM} <: AbstractP4MLBasis
   basis::TB
   transform::TM
   meta::Dict{String, Any}
end 

Base.length(basis::TransformedBasis) = size(basis.transform, 1)

natural_indices(basis::TransformedBasis) = 1:length(basis)

index(basis::TransformedBasis, m::Integer) = m

Base.show(io::IO, basis::TransformedBasis) = 
   print(io, "TransformedBasis(maxn = $(length(basis)), maxq = $(length(basis.basis))")

_valtype(basis::TransformedBasis, TX::Type{T}) where {T} = 
            promote_type(T, eltype(basis.transform))

_generate_input(basis::TransformedBasis) =  
            _generate_input(basis.basis)

# --------------------------------------------------

function evaluate!(Q::AbstractArray, 
                   basis::TransformedBasis, 
                   x::Number)
   P = @withalloc evaluate!(basis.basis, x)
   mul!(Q, basis.transform, P)
   return Q
end

function evaluate_ed!(Q::AbstractArray, dQ::AbstractArray, 
                      basis::TransformedBasis, 
                      x::Number)
   P, dP = @withalloc evaluate_ed!(basis.basis, x)
   mul!(Q, basis.transform, P)
   mul!(dQ, basis.transform, dP)
   return Q, dQ
end

function evaluate_ed2!(Q::AbstractArray, dQ::AbstractArray, ddQ, 
                      basis::TransformedBasis, 
                      x::Number)
   P, dP, ddP = @withalloc evaluate_ed2!(basis.basis, x)
   mul!(Q, basis.transform, P)
   mul!(dQ, basis.transform, dP)
   mul!(ddQ, basis.transform, ddP)
   return Q, dQ, ddQ 
end


function evaluate!(Q::AbstractArray, 
                   basis::TransformedBasis, 
                   x::AbstractVector{<: Number})
   P = @withalloc evaluate!(basis.basis, x)
   mul!(Q, P, transpose(basis.transform))
   return Q
end

function evaluate_ed!(Q::AbstractArray, dQ::AbstractArray, 
                      basis::TransformedBasis, 
                      x::AbstractVector{<: Number})
   P, dP = @withalloc evaluate_ed!(basis.basis, x)
   mul!(Q, P, transpose(basis.transform))
   mul!(dQ, dP, transpose(basis.transform))
   return Q, dQ
end

function evaluate_ed2!(Q::AbstractArray, dQ::AbstractArray, ddQ, 
                      basis::TransformedBasis, 
                      x::AbstractVector{<: Number})
   P, dP, ddP = @withalloc evaluate_ed2!(basis.basis, x)
   mul!(Q, P, transpose(basis.transform))
   mul!(dQ, dP, transpose(basis.transform))
   mul!(ddQ, ddP, transpose(basis.transform))
   return Q, dQ, ddQ 
end

# --------------------------------------------------

function _simple_Hk_weights(k, w0) 
   weights = zeros(k+1) 
   weights[1] = w0
   weights[end] = 1
   return weights
end 

function sobolev_basis(maxn;
                        maxq = maxn, 
                        k = 2, w0 = 0.01,
                        weights = _simple_Hk_weights(k, w0),
                        Nquad = 30 * maxq, 
                        xx = range(-1.0, 1.0, length = Nquad))
   # TODO : the uniform grid should be replaced with a gauss quadrature rule 

   @assert minimum(xx) ≈ -1.0 && maximum(xx) ≈ 1.0
   @assert maxq >= maxn 

   L2basis = Polynomials4ML.legendre_basis(maxq) 
                        
   ∇kP = []
   push!(∇kP, x -> L2basis(x))
   p0 = ∇kP[1] 
   P0 = reduce(hcat, [p0(x) for x in xx])
   G = weights[1] * P0 * P0'

   @show size(G)

   for k = 1:length(weights)-1 
      wk = weights[k+1] 
      pk = x -> ForwardDiff.derivative(∇kP[k], x)
      push!(∇kP, pk)
      Pk = reduce(hcat, [pk(x) for x in xx])
      G += wk * Pk * Pk'
   end

   G /= length(xx)

   # The gramian G encodes the following: 
   # G_ij = <Pi, Pj>_Hk 
   #      = ∫ w0 P_i(x) P_j(x) + w1 P'_i(x) P'_j(x) + ... 
   #                       ... + wk P^(k)_i(x) P^(k)_j(x)   dx
   # Recall that ∫ P_i P_j = δ_ij. So the following eigenvaly problem 
   # solves 
   #     < Vi, u >_Hk = λi < Vi, u >_L2    ∀ u
   # In linear algebra notation, 
   #      G V[:, i] = λi V[:, i]
   λ, V = eigen(Symmetric(G))
   
   # We now define a new basis    Q = V' * P    then 
   #    <Qi, Qj>_Hk = ∑_a,b V_ai V_bj <Pa, Pb>_Hk 
   #                = ∑_a V_ai V_bj G_ab 
   #                = [ V[:, i]' G V[:, j]
   #                = λi δ_ij 
   #
   # This means that the new basis if Hk orthogonal (not orthonormal!) and 
   # λi is a measure for smoothness of Qi. 
   # we store that information in the meta-data of the basis.

   T = collect(V[:, 1:maxn]')
   meta = Dict("info" => "sobolev_basis", 
               "nodes" => xx, 
               "weights" => weights,
               "lambda" => λ[1:maxn])

   return TransformedBasis(L2basis, T, meta) 
end
