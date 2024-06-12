import SpheriCart
import SpheriCart: idx2lm, lm2idx
using SpheriCart: compute, compute_with_gradients, 
				      compute!, compute_with_gradients!, 
						SphericalHarmonics, SolidHarmonics
using LinearAlgebra: norm 

export real_sphericalharmonics, real_solidharmonics, 
		complex_sphericalharmonics, complex_solidharmonics

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
		RealSCWrapper(SolidHarmonics(L; 
						  normalisation = normalisation, static = static, kwargs...))

complex_sphericalharmonics(L; normalisation = :L2, static=false, kwargs...) = 
		ComplexSCWrapper(SphericalHarmonics(L; 
						  normalisation = normalisation, static = static, kwargs...))

complex_solidharmonics(L; normalisation = :L2, static=false, kwargs...) = 
		ComplexSCWrapper(SolidHarmonics(L; 
						  normalisation = normalisation, static = static, kwargs...))

maxl(basis::SCWrapper) = maxl(basis.scbasis)
maxl(scbasis::SphericalHarmonics{L}) where {L} = L
maxl(scbasis::SolidHarmonics{L}) where {L} = L

Base.length(basis::SCWrapper) = SpheriCart.sizeY(maxl(basis))

_generate_input(basis::SCWrapper) = _generate_input(basis.scbasis)

function _generate_input(scbasis::SphericalHarmonics)
	u = @SVector randn(3)
	return u / norm(u)
end

function _generate_input(scbasis::SolidHarmonics)
	u = @SVector randn(3)
	return rand() * (u / norm(u))
end


# ---------------------- Nicer output 

_ℝℂ(::RealSCWrapper) = "ℝ"
_ℝℂ(::ComplexSCWrapper) = "ℂ"

Base.show(io::IO, basis::SCWrapper) =  
		print(io, "$(typeof(basis.scbasis).name.name)($(_ℝℂ(basis)), maxl=$(maxl(basis)))") 

# ---------------------- P4ML Interface stuff 

natural_indices(basis::SCWrapper) = 
      [ NamedTuple{(:l, :m)}(idx2lm(i)) for i = 1:length(basis) ]

_valtype(sh::RealSCWrapper, ::Type{<: StaticVector{3, S}}) where {S} = S

_valtype(sh::ComplexSCWrapper, ::Type{<: StaticVector{3, S}}) where {S} = Complex{S}

#    ::Type{<: StaticVector{3, Hyper{S}}}) where {L, NRM, STATIC, T <: Real, S <: Real} = 
# promote_type(T, Hyper{S})


function evaluate!(Y::AbstractArray, basis::SCWrapper, x::SVector{3})
	Y_temp = reshape(Y, 1, :)
	compute!(Y_temp, basis.scbasis, SA[x,])
	_convert_R2C!(Y, basis)
	return Y
end

function evaluate_ed!(Y::AbstractArray, dY::AbstractArray, 
							 basis::SCWrapper, x::SVector{3})
	Y_temp = reshape(Y, 1, :)
	dY_temp = reshape(dY, 1, :)
	compute_with_gradients!(Y_temp, dY_temp, basis.scbasis, SA[x,])
	_convert_R2C!(Y, basis)
	_convert_R2C!(dY, basis)
	return Y, dY
end

function evaluate!(Y::AbstractArray, 
		    basis::SCWrapper, X::AbstractVector{<: SVector{3}}) 
	compute!(Y, basis.scbasis, X)
	_convert_R2C!(Y, basis)
	return Y 
end

function evaluate_ed!(Y::AbstractArray, dY::AbstractArray, 
			    basis::SCWrapper, X::AbstractVector{<: SVector{3}}) 
	compute_with_gradients!(Y, dY, basis.scbasis, X)
	_convert_R2C!(Y, basis)
	_convert_R2C!(dY, basis)
	return Y, dY
end

# ---------------------- Real to complex conversion 

function _convert_R2C!(Y, basis::RealSCWrapper)
	return Y 
end 

function _convert_R2C!(Y::AbstractVector, basis::ComplexSCWrapper)
	LMAX = maxl(basis)
   for l = 0:LMAX
      # m = 0 => do nothing 
      # m ≠ 0 => linear combinations of ± m terms 
      for m = 1:l 
         i_lm⁺ = SpheriCart.lm2idx(l,  m)
         i_lm⁻ = SpheriCart.lm2idx(l, -m)
         Ylm⁺ = Y[i_lm⁺]
         Ylm⁻ = Y[i_lm⁻]
         Y[i_lm⁺] = (-1)^m * (Ylm⁺ + im * Ylm⁻) / sqrt(2)
         Y[i_lm⁻] = (Ylm⁺ - im * Ylm⁻) / sqrt(2)
      end
   end 
	return Y 
end 


function _convert_R2C!(Y::AbstractMatrix, basis::ComplexSCWrapper)
	LMAX = maxl(basis)
	Nx = size(Y, 1) 
   for l = 0:LMAX
      # m = 0 => do nothing 
      # m ≠ 0 => linear combinations of ± m terms 
      @inbounds  for m = 1:l 
         i_lm⁺ = SpheriCart.lm2idx(l,  m)
         i_lm⁻ = SpheriCart.lm2idx(l, -m)
			@simd ivdep for j = 1:Nx 
         	Ylm⁺ = Y[j, i_lm⁺]
         	Ylm⁻ = Y[j, i_lm⁻]
         	Y[j, i_lm⁺] = (-1)^m * (Ylm⁺ + im * Ylm⁻) / sqrt(2)
         	Y[j, i_lm⁻] = (Ylm⁺ - im * Ylm⁻) / sqrt(2)
			end 
      end
   end 
	return Y 
end 

