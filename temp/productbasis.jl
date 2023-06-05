# Jerry: This is just a specific case of a general ProductBasis
# I will do that later expanding this to a general case, but it is unclear
# to me how to allow the basis to distinguish whether to use norm(x) or x efficiently
using Lux: WrappedFunction
using Lux

struct ProductBasis{NB, TR, TY, TS} <: AbstractPoly4MLBasis
   spec1::Vector{TS}
   bRnl::TR
   bYlm::TY
   # ---- evaluation kernel ---- 
   sparsebasis::SparseProduct{NB}
   @reqfields
end

function _invmap(a::AbstractVector)
   inva = Dict{eltype(a), Int}()
   for i = 1:length(a) 
      inva[a[i]] = i 
   end
   return inva 
end

function dropnames(namedtuple::NamedTuple, names::Tuple{Vararg{Symbol}}) 
   keepnames = Base.diff_names(Base._nt_names(namedtuple), names)
   return NamedTuple{keepnames}(namedtuple)
end

function ProductBasis(spec1, bRnl, bYlm)
   spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1)) 
   spec_Rnl = bRnl.spec; inv_Rnl = _invmap(spec_Rnl)
   spec_Ylm = natural_indices(bYlm); inv_Ylm = _invmap(spec_Ylm)

   spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1))
   for (i, b) in enumerate(spec1)
      spec1idx[i] = (inv_Rnl[dropnames(b,(:m,))], inv_Ylm[(l=b.l, m=b.m)])
   end
   sparsebasis = SparseProduct(spec1idx)
   return ProductBasis(spec1, bRnl, bYlm, sparsebasis, _make_reqfields()...)
end


(pbasis::ProductBasis)(args...) = evaluate(pbasis, args...)

function evaluate(basis::ProductBasis, X::AbstractVector{<: AbstractVector})
   Nel = length(X)
   T = promote_type(eltype(X[1]))
   
   # create all the shifted configurations 
   # for i = 1:Nel
   #    xx[i] = norm(X[i])
   # end
   
   # evaluate the radial and angular components on all the shifted particles 
   Rnl = evaluate(basis.bRnl, (norm.(X))[:])
   Ylm = evaluate(basis.bYlm, X[:])
   
   # evaluate all the atomic orbitals as ϕ_nlm = Rnl * Ylm 
   ϕnlm = evaluate(basis.sparsebasis, (Rnl, Ylm))

   return ϕnlm
end

function ProductBasisLayer(spec1, bRnl, bYlm)
   spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1))
   spec_Rnl = natural_indices(bRnl); inv_Rnl = _invmap(spec_Rnl)
   spec_Ylm = natural_indices(bYlm); inv_Ylm = _invmap(spec_Ylm)

   spec1idx = Vector{Tuple{Int, Int}}(undef, length(spec1))
   for (i, b) in enumerate(spec1)
      spec1idx[i] = (inv_Rnl[dropnames(b,(:m,))], inv_Ylm[(l=b.l, m=b.m)])
   end
   sparsebasis = SparseProduct(spec1idx)

   # wrap into lux layers
   l_Rn = Polynomials4ML.lux(bRnl)
   l_Ylm = Polynomials4ML.lux(bYlm)
   l_ϕnlm = Polynomials4ML.lux(sparsebasis)
   
   # formming model with Lux Chain
   _norm(x) = norm.(x)

   l_xnx = Lux.Parallel(nothing; normx = WrappedFunction(_norm), x = WrappedFunction(identity))
   l_embed = Lux.Parallel(nothing; Rn = l_Rn, Ylm = l_Ylm)
   return Chain(; xnx = l_xnx, embed = l_embed, ϕnlms = l_ϕnlm)
end