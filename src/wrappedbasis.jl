
"""
   struct WrappedBasis 

A wrapper type for a Lux layer `l` that behaves like a P4ML basis. The wrapper 
implements the necessary methods to make `WrappedBasis` a valid `AbstractP4MLBasis`.
It assumes that the following calls are valid: 
- `l(x::T, ps, st)` with `T <: Number` produces an `AbstractVector{T}`; i.e. 
- `l(X::AbstractVector{T}, ps, st)` produces an `B::AbstractMatrix{T}` of 
numbers, with `B[i, :] == l(X[i], ps, st)`.

In particular it is assume that input and output types match. If this 
fails then the behaviour is undefined. (With the non-allocating interface 
this is likely unproblematic. Witht he allocating interface one could 
monkey-patch `_valtype` to get around this restriction.)

The forwardpass is computed via `l(x, ps, st)`. Due to the above assumption, 
the optimal implementation of derivatives is forward-mode, hence `evaluate_ed` is
implemented via `ForwardDiff`, and the rrule is provided by the standard P4ML 
interface. 
"""
struct WrappedBasis{TL} <: AbstractP4MLBasis
   l::TL
   len::Int 
end

function wrapped_basis(l; rng = Random.default_rng(), x = 0.0)
   ps, st = LuxCore.setup(rng, l)
   P, _ = l(x, ps, st)
   if !(isa(P, AbstractVector{<: Number}))
      throw(ArgumentError("The Lux layer `l` does not return a vector of numbers for scalar input."))
   end
   return WrappedBasis{typeof(l)}(l, length(P))
end

LuxCore.initialparameters(rng::AbstractRNG, b::WrappedBasis) = 
      LuxCore.initialparameters(rng, b.l) 

LuxCore.initialstates(rng::AbstractRNG, b::WrappedBasis) = 
      LuxCore.initialstates(rng, b.l) 

_valtype(basis::WrappedBasis, T::Type{TX}) where {TX} = TX 

Base.length(basis::WrappedBasis) = basis.len

# This method for _evaluate! gets around the issue that basis.l can convert 
# a StaticBatch into an SVector, which then messes badly with the 
# type dispatch. It is an extra allocation but since this basis is 
# already allocating anyhow, it is probably not a great loss. Still 
# an unfortunate side-effect...
# TODO: find a better workaround? => probably remove sphericart from P4ml 
#                                   and define P4ML interface only for scalars
_evaluate!(P, dP, basis::WrappedBasis, X::StaticBatch, ps, st) = 
      _evaluate!(P, dP, basis, Vector(X), ps, st)

function _evaluate!(P, dP, basis::WrappedBasis, X::AbstractVector{<: Number}, 
                    ps, st)
   if isnothing(dP) 
      Pl, _ = basis.l(X, ps, st)
      copy!(P, Pl)
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

         Pd, _ = basis.l(Xd, ps, st)

         for i = 1:length(X), j = 1:size(P, 2)
            P[i, j] = Pd[i, j].value # FD.value(Pd[i][j])
            dP[i, j] = Pd[i, j].partials[1]
         end
      end
   end

   return nothing 
end

#
# doesn't need a real kernel launcher since all the actual 
# kernels live inside...
#
function _ka_evaluate_launcher!(P, dP, 
                     basis::WrappedBasis, 
                     X::AbstractVector{<: Number}, 
                     ps, st)

   __dualize(x)= FD.Dual(x, one(eltype(x)))                     
   
   if isnothing(dP) 
      Pl, _ = basis.l(X, ps, st)
      copy!(P, Pl)
   else
      TX = eltype(X)
      TP = _valtype(basis, TX) 
      TXd = typeof(FD.Dual(one(TX), one(TX)))
      Xd = similar(X, TXd)
      map!(__dualize, Xd, X)
      # Xd = map(x -> FD.Dual(x, one(TX)), X)
      Pd, _ = basis.l(Xd, ps, st)

      map!(pd -> pd.value, P, Pd) 
      map!(pd -> pd.partials[1], dP, Pd)
   end

   return nothing 
end
