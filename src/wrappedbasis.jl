
"""
   struct WrappedBasis 

A wrapper type for a Lux layer `l` that behaves like a P4ML basis. The wrapper 
implements the necessary methods to make `LuxBasis` a valid `AbstractP4MLBasis`.
It assumes that the following calls are valid: 
- `l(x::Number, ps, st)` produces an `AbstractVector`
- `l(X::AbstractVector{<:Number}, ps, st)` produces an `B::AbstractMatrix` of 
numbers, with `B[i, :] == l(X[i], ps, st)`.

When used with the allocating interface `evaluate` and `evaluate_ed`, then 
it is also assumed that the input type and output eltype are the same. If this 
is not the case, then one should monkey-patch `_valtype(::LuxBasis, T::Type)`.

The forwardpass is computed via `l(x, ps, st)`. Due to the above assumption, 
the optimal implementation of derivatives is forward-mode, hence `evaluate_ed` is
implemented via `ForwardDiff`, and the rrule is provided by the standard P4ML 
interface. 
"""
struct WrappedBasis{TL, TX, TP} <: AbstractP4MLBasis
   l::TL
   len::Int 
end

function wrapped_basis(l, x; rng = Random.default_rng())
   ps, st = LuxCore.setup(rng, l)
   P, _ = l(x, ps, st)
   if !(isa(P, AbstractVector{<: Number}))
      throw(ArgumentError("The Lux layer `l` does not return a vector of numbers for scalar input."))
   end
   TX = typeof(x) 
   TP = eltype(P)  
   return WrappedBasis{typeof(l), TX, TP}(l, length(P))
end

LuxCore.initialparameters(rng::AbstractRNG, b::WrappedBasis) = 
      LuxCore.initialparameters(rng, b.l) 

LuxCore.initialstates(rng::AbstractRNG, b::WrappedBasis) = 
      LuxCore.initialstates(rng, b.l) 




# function chained_basis()


_valtype(basis::WrappedBasis{TL, TX, TP}, T::Type{TX}
        ) where {TL, TX, TP} = TP 

_valtype(basis::WrappedBasis{TL, TX, TP}, T::Type{Dual{S, TX, TX}}
        ) where {TL, TX, TP, S} = Dual{S, TP, TP} 

# _valtype(basis::WrappedBasis{TL, TX, TP}, T::Type{TX}) = 
#       throw( ArgumentError("WrappedBasis must be initialized with the correct argument type") )

Base.length(basis::WrappedBasis) = basis.len


function _evaluate!(P, dP, basis::WrappedBasis, X::AbstractVector{<: Number}, 
                    ps, st)
   if isnothing(dP) 
      Pl, _ = basis.l(X, ps, st)
      P[:] = Pl
   else
      @no_escape begin 
         TX = eltype(X)
         TP = _valtype(basis, TX) 
         TXd = typeof(FD.Dual(X[1], one(TX)))
         Xd = @alloc(TXd, length(X))
         for i = 1:length(X)
            Xd[i] = FD.Dual(X[i], one(TX))
         end
         
         # if we had an in-place evaluation for l ... 
         # TPd = typeof(FD.Dual(one(TP), one(TP)))
         # Pd = @alloc(TPd, size(P))

         Pd = basis.l(Xd, ps, st)

         for i = 1:length(X), j = 1:size(P, 2)
            P[i, j] = FD.value(Pd[i][j])
            dP[i, j] = Pd[i][j].partials[1]
         end
      end
   end

   return nothing 
end


