import SpheriCart
import SpheriCart: idx2lm, lm2idx
using SpheriCart: compute, compute_with_gradients, 
				      compute!, compute_with_gradients!, 
						SphericalHarmonics, SolidHarmonics

export real_sphericalharmonics, real_solidharmonics						

abstract type SCWrapper <: AbstractP4MLBasis end 

struct RealSCWrapper{SCT} <: SCWrapper
	scbasis::SCT
	@reqfields
end						

struct ComplexSCWrapper{SCT} <: SCWrapper
	scbasis::SCT
	@reqfields
end						


# ---------------------- Convenience constructors & Accessors 

RealSCWrapper(scbasis) = RealSCWrapper(scbasis, _make_reqfields()...)
ComplexSCWrapper(scbasis) = ComplexSCWrapper(scbasis, _make_reqfields()...)

real_sphericalharmonics(L; normalisation = :L2, static=false, kwargs...) = 
		RealSCWrapper(SphericalHarmonics(L; 
						  normalisation = normalisation, static = static, kwargs...))

real_solidharmonics(L; normalisation = :L2, static=false, kwargs...) = 
		RealSCWrapper(SolidHarmonics(L); 
						  normalisation = normalisation, static = static, kwargs...)


maxl(basis::SCWrapper) = maxl(basis.scbasis)
maxl(scbasis::SphericalHarmonics{L}) where {L} = L
maxl(scbasis::SolidHarmonics{L}) where {L} = L

Base.length(basis::SCWrapper) = SpheriCart.sizeY(maxl(basis))


# ---------------------- Nicer output 

_ℝℂ(::RealSCWrapper) = "ℝ"
_ℝℂ(::ComplexSCWrapper) = "ℂ"

Base.show(io::IO, basis::SCWrapper) =  
		print(io, "$(typeof(basis.scbasis).name.name)($(_ℝℂ(basis)), maxl=$(maxl(basis)))") 

# ---------------------- P4ML Interface stuff 

natural_indices(basis::SCYlmBasis) = 
      [ NamedTuple{(:l, :m)}(idx2lm(i)) for i = 1:length(basis) ]

_valtype(sh::RealSCWrapper, ::Type{<: StaticVector{3, S}}) where {S} = S

_valtype(sh::ComplexSCWrapper, ::Type{<: StaticVector{3, S}}) where {S} = Complex{S}

#    ::Type{<: StaticVector{3, Hyper{S}}}) where {L, NRM, STATIC, T <: Real, S <: Real} = 
# promote_type(T, Hyper{S})


function evaluate!(Y::AbstractArray, basis::RealSCWrapper, x::SVector{3})
	Y_temp = reshape(Y, 1, :)
	compute!(Y_temp, basis.scbasis, SA[x,])
	return Y
end

function evaluate_ed!(Y::AbstractArray, dY::AbstractArray, 
							 basis::RealSCWrapper, x::SVector{3})
	Y_temp = reshape(Y, 1, :)
	dY_temp = reshape(dY, 1, :)
	compute_with_gradients!(Y_temp, dY_temp, basis.scbasis, SA[x,])
	return Y, dY
end

evaluate!(Y::AbstractArray, 
		    basis::RealSCWrapper, X::AbstractVector{<: SVector{3}}) = 
	compute!(Y, basis.scbasis, X)

evaluate_ed!(Y::AbstractArray, dY::AbstractArray, 
			    basis::RealSCWrapper, X::AbstractVector{<: SVector{3}}) = 
	compute_with_gradients!(Y, dY, basis.scbasis, X)




# # rrule
# function ChainRulesCore.rrule(::typeof(evaluate), basis::SCYlmBasis, X)
# 	A, dX = evaluate_ed(basis, X)
# 	function pb(∂A)
# 		@assert size(∂A) == (length(X), length(basis))
# 		T∂X = promote_type(eltype(∂A), eltype(dX))
# 		∂X = similar(X, SVector{3, T∂X})
# 		for i = 1:length(X)
#             ∂X[i] = sum([∂A[i,j] * dX[i,j] for j = 1:length(dX[i,:])])
#         end
# 		return NoTangent(), NoTangent(), ∂X
# 	end
# 	return A, pb
# end

=#