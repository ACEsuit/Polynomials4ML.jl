using SpheriCart: compute, compute_with_gradients, compute!, compute_with_gradients!, SphericalHarmonics
export SCYlmBasis

struct SCYlmBasis{L, normalisation, static, T} <: SVecPoly4MLBasis
	basis::SphericalHarmonics{L, normalisation, static, T}
	@reqfields
end

maxL(basis::SCYlmBasis{L, normalisation, static, T}) where {L, normalisation, static, T} = L
Base.length(basis::SCYlmBasis) = sizeY(maxL(basis))

SCYlmBasis(maxL::Integer, T::Type=Float64) = 
      SCYlmBasis(SphericalHarmonics(maxL; normalisation = :L2, static = maxL <= 15, T = T))

SCYlmBasis(scsh::SphericalHarmonics) = 
      SCYlmBasis(scsh, _make_reqfields()...)

natural_indices(basis::SCYlmBasis) = 
      [ NamedTuple{(:l, :m)}(idx2lm(i)) for i = 1:length(basis) ]

_valtype(sh::SCYlmBasis{L, normalisation, static, T}, ::Type{<: StaticVector{3, S}}) where {L, normalisation, static, T <: Real, S <: Real} = 
		promote_type(T, S)

_valtype(sh::SCYlmBasis{L, normalisation, static, T}, ::Type{<: StaticVector{3, Hyper{S}}}) where {L, normalisation, static, T <: Real, S <: Real} = 
		promote_type(T, Hyper{S})

Base.show(io::IO, basis::SCYlmBasis{L, normalisation, static, T}) where {L, normalisation, static, T} = 
      print(io, "SCYlmBasis(L=$L)")	

# ---------------------- Interfaces


evaluate!(Y::AbstractArray, basis::SCYlmBasis, X::SVector{3}) = [Y[i] = compute(basis.basis,X)[i] for i = 1:length(Y)] # scYlm!(Y,compute(basis.basis,X))
evaluate_ed!(Y::AbstractArray, dY::AbstractArray, basis::SCYlmBasis, X::SVector{3}) = scYlm_ed!(Y,dY,compute_with_gradients(basis.basis,X)...)

function scYlm_ed!(Y,dY,val,dval)
    Y .= val
    for i = 1:length(dY)
        dY[i] = dval[i]
    end
    return Y, dY
end

evaluate!(Y::AbstractArray, basis::SCYlmBasis, X::AbstractVector{<: SVector{3}}) = compute!(Y,basis.basis,X)
evaluate_ed!(Y::AbstractArray, dY::AbstractArray, basis::SCYlmBasis, X::AbstractVector{<: SVector{3}}) = compute_with_gradients!(Y,dY,basis.basis,X)

# rrule
function ChainRulesCore.rrule(::typeof(evaluate), basis::SCYlmBasis, X)
	A, dX = evaluate_ed(basis, X)
	function pb(∂A)
		@assert size(∂A) == (length(X), length(basis))
		T∂X = promote_type(eltype(∂A), eltype(dX))
		∂X = similar(X, SVector{3, T∂X})
		for i = 1:length(X)
            ∂X[i] = sum([∂A[i,j] * dX[i,j] for j = 1:length(dX[i,:])])
        end
		return NoTangent(), NoTangent(), ∂X
	end
	return A, pb
end