

"""
   struct TransformedBasis

Basically a three-stage chain, consisting of an input transformation, 
basis evaluation and then output transformation. Constructor: 
```julia
TransformedBasis(transin, basis, transout)
```      
The point of this structure is to provide such a transformed basis (chain) that 
behaves exactly as all other P4ML bases. 

### Comments
- the "natural indices" will simply be `1:len`, where length is the 
  number of transformed basis function. 
- It is assumed that the input transformation and output transformation do not 
  change the number types.
- `_generate_input` is not implemented for general input transforms; to 
implement it for an input transform of type `TIN` one should monkey-patch 
```julia
Polynomials4ML._generate_input(::TransformedBasis{TIN}, T::Type) where {TIN} = ...
```
- `identity` doesn't behave well with Lux, so don't use it as a tranform, 
instead use `nothing`; this will be treated as an identity transform.
"""
struct TransformedBasis{TIN, BAS, TOUT} <: AbstractP4MLBasis 
   transin::TIN 
   basis::BAS
   transout::TOUT
   len::Int
end


function Base.show(io::IO, l::TransformedBasis)
   print(io, "TransformedBasis(...)")
end

Base.length(basis::TransformedBasis) = basis.len

natural_indices(basis::TransformedBasis) = [ (n = n,) for n = 1:length(basis) ]

_valtype(basis::TransformedBasis, T::Type) = 
    _valtype(basis.basis, T)

_generate_input(basis::TransformedBasis{Nothing}) = 
      _generate_input(basis.basis)

# --------------------------------------------------------- 
# Lux stuff

_init_luxparams(rng::AbstractRNG, l::TransformedBasis) = 
      ( transin = _init_luxparams(rng, l.transin),
        basis = _init_luxparams(rng, l.basis),
        transout = _init_luxparams(rng, l.transout) )

_init_luxstate(rng::AbstractRNG, l::TransformedBasis) =
      ( transin = _init_luxparams(rng, l.transin),
        basis = _init_luxparams(rng, l.basis),
        transout = _init_luxparams(rng, l.transout) )
        
_init_luxparams(rng::AbstractRNG, ::Nothing) = NamedTuple() 
_init_luxstate(rng::AbstractRNG, ::Nothing) = NamedTuple() 

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
      z1 = evaluate(tbasis.transin, x[1], ps.transin, st.transin) 
      TZ = typeof(z1)
      Z = @alloc(TZ, nX) 
      @inbounds begin
         Z[1] = z1
         @simd ivdep for i = 2:nX
            Z[i] = evaluate(tbasis.transin, x[i], ps.transin, st.transin)
         end
      end

      # [2] Stage 2 - evaluate the basis 
      Q = @withalloc evaluate!(tbasis.basis, Z, ps.basis, st.basis)
      dQ = nothing 

      # [3] Stage 3 - transform the basis into the output basis 
      _evaluate!(P, dP, tbasis.transout, Q, dQ, ps.transout, st.transout)
   end

   return nothing 
end



# --------------------------------------------------------- 
# KA kernel 
# 





# ------------------------------ 
# auxiliary transforms 

evaluate(f::Function, x::SINGLE, ps, st) = f(x)

evaluate(::Nothing, x::SINGLE, ps, st) = x

function _evaluate!(P, dP, transout::Nothing, Q::AbstractVector, dQ, ps, st)
   for j = 1:size(Q, 2), n = 1:size(Q, 1)
      P[n, j] = Q[n, j]
      isnothing(dP) || (dP[n, j] = dQ[n, j])
   end
   return nothing 
end
