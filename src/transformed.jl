
import ForwardDiff as FD 

"""
   struct TransformedBasis

Basically a two-stage chain, consisting of an input transformation, 
and basis evaluation. Constructor: 
```julia
TransformedBasis(trans, basis)
```      
The point of this structure is to provide a transformed basis that 
behaves exactly as all other P4ML bases. 

### Comments
- the "natural indices" will be the same as for `basis`
- `_generate_input` is not implemented for general input transforms; to 
implement it for an input transform of type `TIN` one should monkey-patch 
```julia
Polynomials4ML._generate_input(::TransformedBasis{TIN}, T::Type) where {TIN} = ...
```
- `_valtype` is implemented but unclear how well it behaves, might be necessary 
to monkey-patch it as well
"""
struct TransformedBasis{TIN, BAS} <: AbstractP4MLBasis 
   trans::TIN
   basis::BAS
end

function Base.show(io::IO, l::TransformedBasis)
   print(io, "TransformedBasis($(l.trans), $(l.basis))")
end

Base.length(l::TransformedBasis) = 
      length(l.basis)

natural_indices(basis::TransformedBasis) = 
      natural_indices(basis.basis)

function _valtype(basis::TransformedBasis, T::Type)
   T1 = _valtype(basis.trans, T) 
   return _valtype(basis.basis, T1)
end

_generate_input(basis::TransformedBasis{Nothing}) = 
      _generate_input(basis.basis)

# --------------------------------------------------------- 
# Lux stuff

_init_luxparams(rng::AbstractRNG, l::TransformedBasis) = 
      ( trans = _init_luxparams(rng, l.trans),
        basis = _init_luxparams(rng, l.basis), )

_init_luxstate(rng::AbstractRNG, l::TransformedBasis) =
      ( trans = _init_luxparams(rng, l.trans),
        basis = _init_luxparams(rng, l.basis),  )

        
_init_luxparams(rng::AbstractRNG, ::Function) = NamedTuple() 
_init_luxstate(rng::AbstractRNG, ::Function) = NamedTuple() 

evaluate(f::Function, x::SINGLE, ps, st) = f(x)

function evaluate_ed(f::Function, x::Number, ps, st)
   y = f(FD.Dual(x)) 
   return FD.value(y), FD.extract_derivative(y)
end

function evaluate_ed(f::Function, x::SVector, ps, st)
   return f(x), FD.gradient(f, x)
end

function _valtype(f::Function, T::Type)
   TT = Base.return_types(f, Tuple{T}) 
   @assert length(TT) == 1 "Function $f should return a single value"
   return TT[1]
end

_init_luxparams(rng::AbstractRNG, ::Nothing) = NamedTuple() 
_init_luxstate(rng::AbstractRNG, ::Nothing) = NamedTuple() 
evaluate(::Nothing, x::SINGLE, ps, st) = x
evaluate_ed(::Nothing, x::SINGLE, ps, st) = x, one(x)
_valtype(::Nothing, T::Type) = T 

# --------------------------------------------------------- 
# CPU SIMD kernel 
# 


function _evaluate!(P, dP::Nothing, 
                    tbasis::TransformedBasis,
                    x::AbstractVector, 
                    ps, st)
   nX = length(x)

   @no_escape begin
      # [1] Stage 1 - transform the inputs 
      z1 = evaluate(tbasis.trans, x[1], ps.trans, st.trans) 
      TZ = typeof(z1)
      Z = @alloc(TZ, nX) 
      @inbounds begin
         Z[1] = z1
         @simd ivdep for i = 2:nX
            Z[i] = evaluate(tbasis.trans, x[i], ps.trans, st.trans)
         end
      end

      # [2] Stage 2 - evaluate the basis 
      evaluate!(P, tbasis.basis, Z, ps.basis, st.basis)
   end

   return nothing 
end



# --------------------------------------------------------- 
# KA kernel 
# 
