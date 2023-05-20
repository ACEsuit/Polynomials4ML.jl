struct SparseProduct{NB} <: AbstractPoly4MLBasis
   spec::Vector{NTuple{NB, Int}}
   # ---- temporaries & caches
   @reqfields()   
end

function SparseProduct()
   return SparseProduct(bases, NTuple{NB, Int}[])
end

# each column defines a basis element
function SparseProduct(spec::Matrix{<: Integer})
   @assert all(spec .> 0)
   spect = [ Tuple(spec[:, i]...) for i = 1:size(spec, 2) ]
   return SparseProduct(spect)
end
 
Base.length(basis::SparseProduct) = length(basis.spec)

SparseProduct(spec) = SparseProduct(spec, _make_reqfields()...)

_valtype(basis::SparseProduct{T1}, TX::NTuple{NB, AbstractVecOrMat{T2}}) where {T1, T2, NB} = T2

# ----------------------- evaluation interfaces 


# function evaluate(basis::SparseProduct, BB::Tuple{Vararg{AbstractVector}}) 
#    VT = mapreduce(eltype, promote_type, BB)
#    A = zeros(VT, length(basis))
#    evaluate!(A, basis, BB::Tuple)
#    return A
# end

# function evaluate(basis::SparseProduct, BB::Tuple{Vararg{AbstractMatrix}}) 
#    VT = mapreduce(eltype, promote_type, BB)
#    nX = size(BB[1], 1)
#    A = zeros(VT, nX, length(basis))
#    evaluate!(A, basis, BB::Tuple)
#    return A 
# end
   
function evaluate_ed(basis::SparseProduct, BB::Tuple{Vararg{AbstractVector}}) 
   VT = mapreduce(eltype, promote_type, BB)
   A = zeros(VT, length(basis))
   _similar(BB::Tuple) = Tuple([similar(BB[i]) for i = 1:length(BB)])
   dA = [_similar(BB) for _ = 1:length(basis)]
   evaluate_ed!(A, dA, basis, BB::Tuple)
   return A, dA
end

function evaluate_ed(basis::SparseProduct, BB::Tuple{Vararg{AbstractMatrix}}) 
   VT = mapreduce(eltype, promote_type, BB)
   nX = size(BB[1], 1)
   A = zeros(VT, nX, length(basis))
   _similar(BB::Tuple) = Tuple([similar(BB[i]) for i = 1:length(BB)])
   dA = [_similar(BB) for i = 1:nX, j = 1:length(basis)] # nX * basis
   evaluate_ed!(A, dA, basis, BB::Tuple)
   return A, dA
end

function evaluate_ed2(basis::SparseProduct, BB::Tuple{Vararg{AbstractVector}}) 
   VT = mapreduce(eltype, promote_type, BB)
   A = zeros(VT, length(basis))
   _similar(BB::Tuple) = Tuple([similar(BB[i]) for i = 1:length(BB)])
   dA, ddA = ([_similar(BB) for _ = 1:length(basis)], [_similar(BB) for _ = 1:length(basis)])
   evaluate_ed2!(A, dA, ddA, basis, BB::Tuple)
   return A, dA, ddA
end

function evaluate_ed2(basis::SparseProduct, BB::Tuple{Vararg{AbstractMatrix}}) 
   VT = mapreduce(eltype, promote_type, BB)
   nX = size(∂∂BB[1], 1)
   A = zeros(VT, nX, length(basis))
   _similar(BB::Tuple) = Tuple([similar(BB[i]) for i = 1:length(BB)])
   dA, ddA = ([_similar(BB) for i = 1:nX, j = 1:length(basis)], [_similar(BB) for i = 1:nX, j = 1:length(basis)])
   evaluate_ed2!(A, dA, ddA, basis, BB::Tuple)
   return A, dA, ddA
end

function _frule_evaluate(basis::SparseProduct, BB::Tuple{Vararg{AbstractVector}}, ∂BB::Tuple{Vararg{AbstractVector}}) 
   VT = mapreduce(eltype, promote_type, BB)
   A = zeros(VT, length(basis))
   # ∂BB: Vector of SVector{3, Float64}
   # dA: Matrix 3 * length(basis)
   dA = zeros(VT, length(∂BB[1][1]), length(basis)) 
   _frule_evaluate!(A, dA, basis, BB::Tuple, ∂BB::Tuple)
   return A, dA
end

function _frule_evaluate(basis::SparseProduct, BB::Tuple{Vararg{AbstractMatrix}}, ∂BB::Tuple{Vararg{AbstractMatrix}}) 
   VT = mapreduce(eltype, promote_type, BB)
   nX = size(∂BB[1], 1)
   # BB: Matrix Nel * length(basis)
   # ∂BB: Matrix of SVector{3, Float64}: Nel * length(basis)
   A = zeros(VT, nX, length(basis))
   dA = [zeros(VT, length(∂BB[1][1])) for i = 1:nX, j = 1:length(basis)]
   _frule_evaluate!(A, dA, basis, BB::Tuple, ∂BB::Tuple)
   return A, dA
end

function _frule_frule_evaluate(basis::SparseProduct, BB::Tuple{Vararg{AbstractVector}}, ∂BB::Tuple{Vararg{AbstractVector}}, ∂∂BB::Tuple{Vararg{AbstractVector}}) 
   VT = mapreduce(eltype, promote_type, BB)
   A = zeros(VT, length(basis))
   # ∂BB: Vector of SVector{3, Float64}
   # dA: Matrix 3 * length(basis)
   dA = zeros(VT, length(∂BB[1][1]), length(basis)) 
   ddA = zeros(VT, length(∂BB[1][1]), length(basis)) 
   _frule_frule_evaluate!(A, dA, ddA, basis, BB::Tuple, ∂BB::Tuple, ∂∂BB::Tuple)
   return A, dA, ddA
end

function _frule_frule_evaluate(basis::SparseProduct, BB::Tuple{Vararg{AbstractMatrix}}, ∂BB::Tuple{Vararg{AbstractMatrix}}, ∂∂BB::Tuple{Vararg{AbstractMatrix}}) 
   VT = mapreduce(eltype, promote_type, BB)
   nX = size(∂BB[1], 1)
   # BB: Matrix Nel * length(basis)
   # ∂BB: Matrix of SVector{3, Float64}: Nel * length(basis)
   A = zeros(VT, nX, length(basis))
   dA = [zeros(VT, length(∂BB[1][1])) for i = 1:nX, j = 1:length(basis)]
   ddA = [zeros(VT, length(∂BB[1][1])) for i = 1:nX, j = 1:length(basis)]
   _frule_frule_evaluate!(A, dA, ddA, basis, BB::Tuple, ∂BB::Tuple, ∂∂BB::Tuple)
   return A, dA, ddA
end
# ----------------------- evaluation kernels 

function evaluate!(A, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractVector}}) where {NB}
   @assert length(BB) == NB
   spec = basis.spec
   for (iA, ϕ) in enumerate(spec)
       @inbounds A[iA] = BB_prod(ϕ, BB)
   end
   return A 
end

function evaluate!(A, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractMatrix}}) where {NB}
   nX = size(BB[1], 1)
   @assert all(B->size(B, 1) == nX, BB)
   spec = basis.spec

   @inbounds for (iA, ϕ) in enumerate(spec)
      @simd ivdep for j = 1:nX
         A[j, iA] = BB_prod(ϕ, BB, j)
      end
   end
   return A
end

# Not sure whether we can everything below
# faster by eval and diff at the same time from prod_grad

function evaluate_ed!(A, dA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractVector}}) where {NB}
   @assert length(BB) == NB
   spec = basis.spec
   # evaluate!(A, basis, BB)
   for (iA, ϕ) in enumerate(spec)
      b = ntuple(Val(NB)) do i 
         @inbounds BB[i][ϕ[i]] 
      end 
      g = _prod_ed(b, Val(NB))
      A[iA] = g[1]
      fill!.(dA[iA], 0.0)
      for i = 1:NB
         dA[iA][i][ϕ[i]] += g[i + 1]
      end
   end 
   return A, dA 
end

function evaluate_ed!(A, dA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractMatrix}}) where {NB}
   nX = size(BB[1], 1)
   @assert all(B->size(B, 1) == nX, BB)
   spec = basis.spec
   # evaluate!(A, basis, BB)
   @inbounds for (iA, ϕ) in enumerate(spec)
      @simd ivdep for j = 1:nX 
        b = ntuple(Val(NB)) do i 
           @inbounds BB[i][j, ϕ[i]] 
        end 
        g = _prod_ed(b, Val(NB))
        A[j, iA] = g[1] 
        fill!.(dA[j, iA], 0.0)
        for i = 1:NB
           dA[j, iA][i][j, ϕ[i]] += g[i + 1]
        end
      end 
   end
   return A, dA
end

function evaluate_ed2!(A, dA, ddA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractVector}}) where {NB}
   @assert length(BB) == NB
   spec = basis.spec

   for (iA, ϕ) in enumerate(spec)
      b = ntuple(Val(NB)) do i 
         @inbounds BB[i][ϕ[i]] 
      end 
      g = _prod_ed2(b, Val(NB))
      A[iA] = g[1]
      fill!.(dA[iA], 0.0)
      fill!.(ddA[iA], 0.0)
      for i = 1:NB 
         dA[iA][i][ϕ[i]] += g[i + 1]
      end
   end 
   return A, dA, ddA 
end


function evaluate_ed2!(A, dA, ddA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractMatrix}}) where {NB}
   nX = size(BB[1], 1)
   @assert all(B->size(B, 1) == nX, BB)
   spec = basis.spec
   # evaluate!(A, basis, BB)
   @inbounds for (iA, ϕ) in enumerate(spec)
      @simd ivdep for j = 1:nX 
        b = ntuple(Val(NB)) do i 
           @inbounds BB[i][j, ϕ[i]] 
        end 
        g = _prod_ed(b, Val(NB))
        A[j, iA] = g[1] 
        fill!.(dA[j, iA], 0.0)
        fill!.(ddA[j, iA], 0.0)
        for i = 1:NB
           dA[j, iA][i][j, ϕ[i]] += g[i + 1]
        end
      end 
   end
   return A, dA
end

function _frule_evaluate!(A, dA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractVector}}, ∂BB::Tuple{Vararg{AbstractVector}}) where {NB}
   @assert length(BB) == NB
   @assert length(∂BB) == NB
   spec = basis.spec
   # evaluate!(A, basis, BB)
   for (iA, ϕ) in enumerate(spec)
      b = ntuple(Val(NB)) do i 
         @inbounds BB[i][ϕ[i]] 
      end 
      g = _prod_ed(b, Val(NB))
      A[iA] = g[1]
      for i = 1:NB
         for j = 1:length(∂BB[1][1])
            dA[j, iA] = muladd(∂BB[i][ϕ[i]][j], g[i + 1], dA[iA])
         end
      end
   end 
   return A, dA 
end

function _frule_evaluate!(A, dA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractMatrix}}, ∂BB::Tuple{Vararg{AbstractMatrix}}) where {NB}
   nX = size(BB[1], 1)
   @assert all(B->size(B, 1) == nX, BB)
   @assert all(∂B->size(∂B, 1) == nX, ∂BB)
   spec = basis.spec
   # evaluate!(A, basis, BB)
   @inbounds for (iA, ϕ) in enumerate(spec)
      @simd ivdep for j = 1:nX 
        b = ntuple(Val(NB)) do i 
           @inbounds BB[i][j, ϕ[i]] 
        end 
        g = _prod_ed(b, Val(NB))
        A[j, iA] = g[1] 
        for i = 1:NB
            for k = 1:length(∂BB[1][1])
               dA[j, iA][k] = muladd(∂BB[i][j, ϕ[i]][k], g[i + 1], dA[j, iA])
            end
        end
      end 
   end
   return A, dA
end

function _frule_frule_evaluate!(A, dA, ddA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractVector}}, ∂BB::Tuple{Vararg{AbstractVector}}, ∂∂BB::Tuple{Vararg{AbstractVector}}) where {NB}
   @assert length(BB) == NB
   @assert length(∂BB) == NB
   @assert length(∂∂BB) == NB
   spec = basis.spec

   for (iA, ϕ) in enumerate(spec)
      b = ntuple(Val(NB)) do i 
         @inbounds BB[i][ϕ[i]] 
      end 
      g = _prod_ed2(b, Val(NB))
      A[iA] = g[1]
      for i = 1:NB 
         for j = 1:length(∂BB[1][1])
            dA[iA, j] = muladd(∂BB[i][ϕ[i]][j], g[i + 1], dA[iA])
            ddA[iA, j] = muladd(∂∂BB[i][ϕ[i]][j], g[i + 1], ddA[iA])
         end
      end
      t = 1
      for m = 1:NB-1
         for n = m+1:NB
            for j = 1:length(∂BB[1][1])
               ddA[iA, j] = muladd(2 * ∂BB[m][ϕ[m]][j] * ∂BB[n][ϕ[n]][j], g[t + 1 + NB], ddA[iA])
            end
            t += 1
         end
      end
   end 
   return A, dA, ddA 
end

function _frule_frule_evaluate!(A, dA, ddA, basis::SparseProduct{NB}, BB::Tuple{Vararg{AbstractMatrix}}, ∂BB::Tuple{Vararg{AbstractMatrix}}, ∂∂BB::Tuple{Vararg{AbstractMatrix}}) where {NB}
   nX = size(BB[1], 1)
   @assert all(B->size(B, 1) == nX, BB)
   @assert all(∂B->size(∂B, 1) == nX, ∂BB)
   @assert all(∂∂B->size(∂∂B, 1) == nX, ∂∂BB)
   spec = basis.spec
   
   @inbounds for (iA, ϕ) in enumerate(spec)
      @simd ivdep for j = 1:nX
         b = ntuple(Val(NB)) do i 
            @inbounds BB[i][j, ϕ[i]] 
         end 
         g = _prod_ed2(b, Val(NB))
         A[j, iA] = g[1]
         for i = 1:NB 
            for k = 1:length(∂BB[1][1])
               dA[j, iA][k] = muladd(∂BB[i][j, ϕ[i]][k], g[i + 1], dA[j, iA])
               ddA[j, iA][k] = muladd(∂∂BB[i][j, ϕ[i]][k], g[i + 1], ddA[j, iA])
            end
         end
         t = 1
         for m = 1:NB-1
            for n = m+1:NB
               ddA[j, iA][k] = muladd(2 * ∂BB[m][j, ϕ[m]][k] * ∂BB[n][j, ϕ[n]][k], g[t + 1 + NB], ddA[j, iA])
            end
            t += 1
         end
      end
   end
   return A, dA, ddA
end
# -------------------- reverse mode gradient

function _rrule_evaluate(basis::SparseProduct{NB}, BB::Tuple) where {NB}
   A = evaluate(basis, BB)
   return A, ∂A -> _pullback_evaluate(∂A, basis, BB)
end


function _pullback_evaluate(∂A, basis::SparseProduct{NB}, BB::Tuple) where {NB}
   TA = promote_type(eltype.(BB)...)
   ∂BB = ntuple(i -> zeros(TA, size(BB[i])...), NB)
   _pullback_evaluate!(∂BB, ∂A, basis, BB)
   return ∂BB
end


function _pullback_evaluate!(∂BB, ∂A, basis::SparseProduct{NB}, BB::Tuple) where {NB}
   nX = size(BB[1], 1)

   @assert all(nX <= size(BB[i], 1) for i = 1:NB)
   @assert all(nX <= size(∂BB[i], 1) for i = 1:NB)
   @assert all(size(∂BB[i], 2) >= size(BB[i], 2) for i = 1:NB)
   @assert size(∂A) == (nX, length(basis))
   @assert length(BB) == NB 
   @assert length(∂BB) == NB
   
   @inbounds for (iA, ϕ) in enumerate(basis.spec) # for each spec
      # ∂A_iA = ∂A[iA]
      @simd ivdep for j = 1:nX 
        b = ntuple(Val(NB)) do i 
           @inbounds BB[i][j, ϕ[i]] 
        end 
        g = _prod_grad(b, Val(NB))
        for i = 1:NB 
           ∂BB[i][j, ϕ[i]] = muladd(∂A[j, iA], g[i], ∂BB[i][j, ϕ[i]])
        end
      end 
   end
   return nothing 
end

test_evaluate(basis::SparseProduct, BB::Tuple) = 
       [ prod(BB[j][basis.spec[i][j]] for j = 1:length(BB)) 
            for i = 1:length(basis) ]

function test_evaluate_ed(basis, BB)
    A = evaluate_ed(basis, BB)[1]
    dA = evaluate_ed(basis, BB)[2]
    errors = Float64[]
    # loop through finite-difference step-lengths
    @printf("---------|----------- \n")
    @printf("    h    | error \n")
    @printf("---------|----------- \n")
    for p = 2:11
        h = 0.1^p
        dAh = deepcopy(dA)
        Δ = deepcopy(dA)
        for n = 1:length(dAh) # basis
            for i = 1:length(dAh[n]) #NB
                for j = 1:length(dAh[n][i]) #BB[i]
                    BB[i][j] += h
                    dAh[n][i][j] = (evaluate(basis, BB)[n] - A[n])/h
                    Δ[n][i][j] = dA[n][i][j] - dAh[n][i][j]
                    BB[i][j] -= h
                end
            end
        end
        push!(errors, maximum([norm(Δ[i][j], Inf) for i = 1:length(Δ), j = 1:length(Δ[i])] ))
        @printf(" %1.1e | %4.2e  \n", h, errors[end])
    end
    @printf("---------|----------- \n")
    if minimum(errors) <= 1e-3 * maximum(errors)
        println("passed")
        return true
   else
        @warn("""It seems the finite-difference test has failed, which indicates
        that there is an inconsistency between the function and gradient
        evaluation. Please double-check this manually / visually. (It is
        also possible that the function being tested is poorly scaled.)""")
        return false
   end
end

function test_evaluate_ed2(basis, BB)
   A = evaluate_ed2(basis, BB)[1]
   ddA = evaluate_ed2(basis, BB)[3]
   errors = Float64[]
   # loop through finite-difference step-lengths
   @printf("---------|----------- \n")
   @printf("    h    | error \n")
   @printf("---------|----------- \n")
   for p = 2:11
       h = 0.1^p
       ddAh = deepcopy(ddA)
       Δ = deepcopy(ddA)
       for n = 1:length(ddAh) # basis
           for i = 1:length(ddAh[n]) #NB
               for j = 1:length(ddAh[n][i]) #BB[i]
                   BB[i][j] += h
                   ddAh[n][i][j] = evaluate(basis, BB)[n] - 2 * A[n]
                   BB[i][j] -= 2*h
                   ddAh[n][i][j] = (ddAh[n][i][j] + evaluate(basis, BB)[n])/h^2
                   BB[i][j] += h 
                   Δ[n][i][j] = ddA[n][i][j] - ddAh[n][i][j]
               end
           end
       end
       push!(errors, maximum([norm(Δ[i][j], Inf) for i = 1:length(Δ), j = 1:length(Δ[i])] ))
       @printf(" %1.1e | %4.2e  \n", h, errors[end])
   end
   @printf("---------|----------- \n")
   if minimum(errors) <= 1e-3 * maximum(errors)
       println("passed")
       return true
  else
       @warn("""It seems the finite-difference test has failed, which indicates
       that there is an inconsistency between the function and gradient
       evaluation. Please double-check this manually / visually. (It is
       also possible that the function being tested is poorly scaled.)""")
       return false
  end
end

