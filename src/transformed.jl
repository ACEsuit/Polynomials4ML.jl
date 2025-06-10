

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
- `_generate_input` is not implemented; to implement it for an input transform 
of type `TIN` one should monkey-patch 
```julia
Polynomials4ML._generate_input(::TransformedBasis{TIN}, T::Type) where {TIN} = ...
```
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

# _generate_input(basis::TransformedBasis) = 

# --------------------------------------------------------- 
# Lux stuff



# --------------------------------------------------------- 
# CPU SIMD kernel 
# 

