
import Interpolations
import ForwardDiff

# TODO: 
#  - consider allowing a coordinate transformation before 
#    applying the splines and a post-multiplication with 
#    an envelope. That's a bit specific for a general purpose 
#    libarary but might be very useful for performance and 
#    and simplicity in ACE applications. 
#    But it might be better to implement these things as wrappers 
#    around an arbitrary basis. 
#
#  - the splines should inherit the specs from the original basis 
#    they interpolated. 


"""
`struct CubicSplines`: 

Statically typed cubic splines, compatible with P4ML type batched evaluation. 
For any P4ML basis with univariate input. 
"""
struct CubicSplines{NX, NU, T} <: AbstractP4MLBasis
   F::SVector{NX, SVector{NU, T}}  # function values at nodes 
   G::SVector{NX, SVector{NU, T}}  # gradient values at nodes 
   x0::T    # left endpoint 
   x1::T    # right endpoint
end 


function Base.show(io::IO, l::CubicSplines{NX, NU, T}) where {NX, NU, T}
   print(io, "CubicSplines(nx = $NX, len = $NU)")
end

Base.length(basis::CubicSplines{NX, NU}) where {NX, NU} = NU

__NX(basis::CubicSplines{NX}) where {NX} = NX

# TODO: this is wrong, instead should inherit from the original basis 
natural_indices(basis::CubicSplines) = [ (n = n,) for n = 1:length(basis) ]

_valtype(basis::CubicSplines{NX, NU, T1}, T2::Type{<:Real}
         ) where {NX, NU, T1} = promote_type(T1, T2)

_valtype(basis::CubicSplines{NX, NU, T1}, T2::Type{<:Real}, 
         ps, st) where {NX, NU, T1} = promote_type(T2, eltype(eltype(st.F[1])))

_generate_input(basis::CubicSplines) = 
         rand() * (basis.x1 - basis.x0) + basis.x0

_init_luxstate(l::CubicSplines) = 
         (F = l.F, G = l.G, x0 = l.x0, x1 = l.x1) 


# ----------------- constructor of spline basis

"""
   splinify(basis, x0, x1, NX; bspline=true)

Takes a P4ML basis with univariate input and constructs a cubic spline basis 
that interpolates the basis functions on a uniform grid with `NX` nodes. 
If `bspline=true` (default) the function values are first interpolated onto
a B-spline representation to obtain C2,2 regularity of the splines. 

`x0`, `x1` are the left and right endpoints of the spline interval. 

This is currently not exported and not part of the public interface. The 
interface can change in future releases.
"""
function splinify(f, x0, x1, NX; bspline=true)
   NU = length(f(x0))
   xx = range(x0, x1; length=NX)
   F = [ SVector{NU}(f(x)) for x in xx ]

   if bspline
      # CubicSplines represents the splines in terms of piecewise cubics 
      # specified through values and gradients at the nodes this only gives C1,1 
      # regularity. By first interpolating onto a B-spline, and then the B-spline 
      # onto the CubicSplines representation we get C2,2 regularity.
      itp = Interpolations.cubic_spline_interpolation(xx, F; 
                     extrapolation_bc=Interpolations.Flat())
      G = [ Interpolations.gradient(itp, x)[1] for x in xx ]
   else 
      G = [ FD.derivative(f, x) for x in xx ]
   end
   T = eltype(eltype(F))
   stF = SVector{NX}(F)
   stG = SVector{NX}((SVector{NU, T}).(G))
   return CubicSplines(stF, stG, x0, x1)
end



# ----------------- shared evaluation code 

"""
   _eval_cubic(t, fl, fr, gl, gr, h)

Evaluate cubic spline at position `t` in `[0,1]`, given function values `fl`, `fr`
and gradients `gl`, `gr` at the left and right endpoints.
"""
@inline function _eval_cubic(t, fl, fr, gl, gr)
   # (2t³ - 3t² + 1)*fl + (t³ - 2t² + t)*gl + 
   #           (-2t³ + 3t²)*fr + (t³ - t²)*gr 
   a0 = fl
   a1 = gl 
   a2 = -3fl + 3fr - 2gl - gr
   a3 = 2fl - 2fr + gl + gr
   return ((a3*t + a2)*t + a1)*t + a0
end

"""
   _eval_cubspl(x, F, G, x0, x1, NX)

auxiliary function to the evaluate the cubic spline basis given 
the spline data arrays    
"""
@inline function _eval_cubspl(x, F, G, x0, x1, NX)
   x = clamp(x, x0, x1)     # project to [x0, x1] (corresponds to Flat bc)
   h = (x1 - x0) / (NX-1)   # uniform grid spacing 
   il = floor(Int, (x - x0) / h)   # index of left node
   # TODO: is this numerically stable? 
   t = (x - x0) / h - il          # relative coordinate of x in [il, il+1]
   @inbounds _eval_cubic(t, F[il+1], F[il+2], h*G[il+1], h*G[il+2])
end

@inline function _cubspl_widthgrad(x, F, G, x0, x1, NX)
   if x < x0 || x > x1
      f = _eval_cubspl(x, F, G, x0, x1, NX)
      return f, zero(f) 
   end
   h = (x1 - x0) / (NX-1)   # uniform grid spacing 
   t, _il = modf((x - x0) / h)
   il = Int(_il)
   td = Dual(t, one(t))
   fd = _eval_cubic(td, F[il+1], F[il+2], h*G[il+1], h*G[il+2])
   f = ForwardDiff.value.(fd)
   g = ForwardDiff.partials.(fd, 1)
   return f, g / h 
end


# ----------------- CPU evaluation code 

_evaluate!(P, dP, basis::CubicSplines, X) = 
      _evaluate!(P, dP, basis, X, nothing, _init_luxstate(basis))


function _evaluate!(P::AbstractMatrix, dP::Nothing, basis::CubicSplines, X::BATCH, ps, st)
   @assert size(P, 1) >= length(X) 
   @inbounds for (i, x) in enumerate(X)
      P[i, :] = _eval_cubspl(x, st.F, st.G, st.x0, st.x1, __NX(basis))
   end
   return nothing 
end 


function _evaluate!(P::AbstractMatrix, dP::AbstractMatrix, 
                    basis::CubicSplines, X::BATCH, ps, st)
   @assert size(P, 1) >= length(X) 
   @assert size(dP, 1) >= length(X) 
   @inbounds for (i, x) in enumerate(X)
      f, g = _cubspl_widthgrad(x, st.F, st.G, st.x0, st.x1, __NX(basis))
      P[i, :] = f
      dP[i, :] = g
   end
   return nothing 
end


# ----------------- KernelAbstractions evaluation code


@kernel function _ka_evaluate!(P, dP, basis::CubicSplines, x::AbstractVector{T}
         ) where {T}

   i = @index(Global)

   if isnothing(dP) 
         P[i, :] = _eval_cubspl(x[i], basis.F, basis.G, basis.x0, basis.x1, __NX(basis))
   else 
      f, g = _cubspl_widthgrad(x[i], basis.F, basis.G, basis.x0, basis.x1, __NX(basis))
      P[i, :] = f
      dP[i, :] = g
   end

   nothing 
end