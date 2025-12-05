
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
end						

struct ComplexSCWrapper{SCT} <: SCWrapper
	scbasis::SCT
end			

_init_luxstate(l::SCWrapper) = (Flm = deepcopy(l.scbasis.Flm),)

# ---------------------- Convenience constructors & Accessors 


"""
`real_sphericalharmonics(L; kwargs...)`

Generate a real spherical harmonics basis (wrapper of sphericart implementation)
"""
real_sphericalharmonics(L; normalisation = :L2, static=false, kwargs...) = 
		RealSCWrapper(SphericalHarmonics(L; 
						  normalisation = normalisation, static = static, kwargs...))

"""
`real_solidharmonics(L; kwargs...)`

Generate a real solid harmonics basis (wrapper of sphericart implementation)
"""
real_solidharmonics(L; normalisation = :L2, static=false, kwargs...) = 
		RealSCWrapper(SolidHarmonics(L; 
						  normalisation = normalisation, static = static, kwargs...))

"""
`complex_sphericalharmonics(L; kwargs...)`

Generate a complex spherical harmonics basis (wrapper of sphericart implementation)
"""
complex_sphericalharmonics(L; normalisation = :L2, static=false, kwargs...) = 
		ComplexSCWrapper(SphericalHarmonics(L; 
						  normalisation = normalisation, static = static, kwargs...))

"""
`complex_solidharmonics(L; kwargs...)`

Generate a complex solid harmonics basis (wrapper of sphericart implementation)
"""
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
	return sqrt(rand()) * (u / norm(u))
end

# ---------------------- Nicer output 

_ℝℂ(::RealSCWrapper) = "ℝ"
_ℝℂ(::ComplexSCWrapper) = "ℂ"

Base.show(io::IO, basis::SCWrapper) =  
		print(io, "$(typeof(basis.scbasis).name.name)($(_ℝℂ(basis)), maxl=$(maxl(basis)))") 

# ---------------------- P4ML Interface stuff 

natural_indices(basis::SCWrapper) = 
      [ NamedTuple{(:l, :m)}(idx2lm(i)) for i = 1:length(basis) ]

# TODO: should the output type only depend on the input type or also 
#       on the type of the Flm parameters? 		

_valtype(sh::RealSCWrapper, ::Type{<: SVector{3, S}}) where {S} = S

_valtype(sh::ComplexSCWrapper, ::Type{<: SVector{3, S}}) where {S} = Complex{S}

# NB: the args... in each of the calls below stands for calls with or without 
#     ps, st. It ought to be possible to just use the generic interface here 
#     this should be looked into. 

function evaluate!(Y::AbstractArray, basis::SCWrapper, x::SVector{3}, args...)
	Y_temp = reshape(Y, 1, :)
	compute!(Y_temp, basis.scbasis, SA[x,])
	_convert_R2C!(Y, basis)
	return Y
end

function evaluate_ed!(Y::AbstractArray, dY::AbstractArray, 
							 basis::SCWrapper, x::SVector{3}, args...)
	Y_temp = reshape(Y, 1, :)
	dY_temp = reshape(dY, 1, :)
	compute_with_gradients!(Y_temp, dY_temp, basis.scbasis, SA[x,])
	_convert_R2C!(Y, basis)
	_convert_R2C!(dY, basis)
	return Y, dY
end

function evaluate!(Y::AbstractArray, 
		    basis::SCWrapper, X::AbstractVector{<: SVector{3}}, args...) 
	compute!(Y, basis.scbasis, X)
	_convert_R2C!(Y, basis)
	return Y 
end

function evaluate!(Y::AbstractGPUArray, 
		    basis::RealSCWrapper, X::AbstractVector{<: SVector{3}}, args...) 
	_ka_evaluate_launcher!(Y, nothing, basis, X, args...)
	return Y 
end



function evaluate_ed!(Y::AbstractArray, dY::AbstractArray, 
			    basis::SCWrapper, X::AbstractVector{<: SVector{3}}, args...) 
	compute_with_gradients!(Y, dY, basis.scbasis, X)
	_convert_R2C!(Y, basis)
	_convert_R2C!(dY, basis)
	return Y, dY
end

function evaluate_ed!(Y::AbstractGPUArray, dY::AbstractGPUArray, 
			    basis::SCWrapper, X::AbstractVector{<: SVector{3}}, args...) 
	_ka_evaluate_launcher!(Y, dY, basis, X, args...)
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

# ---------------------- KernelAbstractions Interface
#
# only for real solid and complex harmonics, since the complex ones aren't 
# supported by the KA kernel in SpheriCart yet. 
# this is a bit of a hack really and we need to iterate on with SC on 
# getting this right. 

_ka_evaluate_launcher!(P, dP, basis::RealSCWrapper, x) = 
			_ka_evaluate_launcher!(P, dP, basis, x, 
										  NamedTuple(), 
										  (Flm = basis.scbasis.Flm,) )


function _ka_evaluate_launcher!(P, dP, 
									basis::RealSCWrapper, 
									x, ps, st)
	nX = length(x) 
	len_basis = length(basis)
	
	@assert size(P, 1) >= nX 
	@assert size(P, 2) >= len_basis 
	if !isnothing(dP)
		@assert size(dP, 1) >= nX
		@assert size(dP, 2) >= len_basis
	end

	_valSH(::SphericalHarmonics) = Val{true}() 
	_valSH(::SolidHarmonics) = Val{false}()

	Flm = st.Flm
	valL = Val{maxl(basis)}()
	valSH = _valSH(basis.scbasis)

	SpheriCart.ka_solid_harmonics!(P, dP, valL, valSH, x, Flm)
	
	return nothing 
end
