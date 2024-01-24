using SpheriCart: compute, compute_with_gradients, 
				      compute!, compute_with_gradients!, 
						SolidHarmonics

export SCRRlmBasis

struct SCRRlmBasis{L, normalisation, static, T} <: SVecPoly4MLBasis
	basis::SolidHarmonics{L, normalisation, static, T}
	@reqfields
end

maxL(basis::SCRRlmBasis{L}) where {L} = L

Base.length(basis::SCRRlmBasis) = sizeY(maxL(basis))

SCRRlmBasis(maxL::Integer, T::Type=Float64) = 
      SCRRlmBasis( SolidHarmonics(maxL; normalisation = :sphericart, 
											          static = maxL <= 15, 
														 T = T))

SCRRlmBasis(srrsh::SolidHarmonics) = 
      SCRRlmBasis(srrsh, _make_reqfields()...)

natural_indices(basis::SCRRlmBasis) = 
      [ NamedTuple{(:l, :m)}(idx2lm(i)) for i = 1:length(basis) ]

_valtype(sh::SCRRlmBasis{L, NRM, STATIC, T}, 
		   ::Type{<: StaticVector{3, S}}) where {L, NRM, STATIC, T <: Real, S <: Real} = 
		promote_type(T, S)

_valtype(sh::SCRRlmBasis{L, NRM, STATIC, T}, 
		   ::Type{<: StaticVector{3, Hyper{S}}}) where {L, NRM, STATIC, T <: Real, S <: Real} = 
		promote_type(T, Hyper{S})

Base.show(io::IO, basis::SCRRlmBasis{L, NRM, STATIC, T}) where {L, NRM, STATIC, T} = 
      print(io, "SCRRlmBasis(L=$L)")	

# ---------------------- Interfaces

function evaluate!(Y::AbstractArray, basis::SCRRlmBasis, X::SVector{3})
	Y_temp = reshape(Y, 1, :)
	compute!(Y_temp, basis.basis, SA[X,])
	return Y
end

function evaluate_ed!(Y::AbstractArray, dY::AbstractArray, basis::SCRRlmBasis, X::SVector{3})
	Y_temp = reshape(Y, 1, :)
	dY_temp = reshape(dY, 1, :)
	compute_with_gradients!(Y_temp, dY_temp, basis.basis, SA[X,])
	return Y, dY
end

evaluate!(Y::AbstractArray, 
		    basis::SCRRlmBasis, X::AbstractVector{<: SVector{3}}) = 
	compute!(Y, basis.basis, X)

evaluate_ed!(Y::AbstractArray, dY::AbstractArray, 
			    basis::SCRRlmBasis, X::AbstractVector{<: SVector{3}}) = 
	compute_with_gradients!(Y, dY, basis.basis, X)

# rrule
function ChainRulesCore.rrule(::typeof(evaluate), basis::SCRRlmBasis, X)
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