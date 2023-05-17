#
# Ordering of the embedding 
# nuc | 1 2 3  1 2 3  1 2 3
#   k | 1 1 1  2 2 2  2 2 2
#
"""
This constructs the specification of all the atomic orbitals for one
nucleus. 

* bRnl : radial basis 
* Ylm : angular basis, assumed to be spherical harmonics 
* admissible : a filter, default is a total degree 
"""
function make_nlms_spec(bRnl, bYlm;
            totaldegree::Integer = -1,
            admissible = ( (br, by) -> degree(bRnl, br) +
                           degree(bYlm, by) <= totaldegree), 
            nnuc = 0)
   
   spec_Rnl = natural_indices(bRnl) # copy(basis.spec)
   spec_Ylm = natural_indices(bYlm) 
   
   spec1 = []
   for (iR, br) in enumerate(spec_Rnl), (iY, by) in enumerate(spec_Ylm)
      if br.l != by.l 
         continue 
      end
      if admissible(br, by)
         push!(spec1, (br..., m = by.m))
      end
   end
   return spec1 
end


# Jerry: This is just a specific case of a general ProductBasis, this should go to Polynomials4ML later with a general implementation
# I will do that after reconfiming this is what we want
mutable struct ProductBasis{NB, TR, TY, TS}
   spec1::Vector{TS}
   bRnl::TR
   bYlm::TY
   # ---- evaluation kernel from Polynomials4ML ---- 
   sparsebasis::SparseProduct{NB}
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
   return ProductBasis(spec1, bRnl, bYlm, sparsebasis)
end


function evaluate(basis::ProductBasis, X::AbstractVector{<: AbstractVector})
   Nel = length(X)
   T = promote_type(eltype(X[1]))
   VT = SVector{3, T}
   
   # create all the shifted configurations 
   xx = zeros(eltype(VT), Nel)
   for i = 1:Nel
      xx[i] = norm(X[i])
   end

   # evaluate the radial and angular components on all the shifted particles 
   Rnl = reshape(evaluate(basis.bRnl, xx[:]), (Nel, length(basis.bRnl)))
   Ylm = reshape(evaluate(basis.bYlm, X[:]), (Nel, length(basis.bYlm)))

   # evaluate all the atomic orbitals as ϕ_nlm = Rnl * Ylm 
   ϕnlm = evaluate(basis.sparsebasis, (Rnl, Ylm))

   return ϕnlm
end