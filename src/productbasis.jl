# Jerry: This is just a specific case of a general ProductBasis
# I will do that later expanding this to a general case, but it is unclear
# to me how to allow the basis to distinguish whether to use norm(x) or x efficiently
struct ProductBasis{NB, TR, TY, TS}
   spec1::Vector{TS}
   bRnl::TR
   bYlm::TY
   # ---- evaluation kernel ---- 
   sparsebasis::SparseProduct{NB}
   @reqfields
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
   Rnl = reshape(evaluate(basis.bRnl, (norm.(X))[:]), (Nel, length(basis.bRnl)))
   Ylm = reshape(evaluate(basis.bYlm, X[:]), (Nel, length(basis.bYlm)))

   # evaluate all the atomic orbitals as ϕ_nlm = Rnl * Ylm 
   ϕnlm = evaluate(basis.sparsebasis, (Rnl, Ylm))

   return ϕnlm
end


