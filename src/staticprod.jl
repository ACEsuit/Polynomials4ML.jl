@inline function BB_prod(ϕ::NTuple{NB}, BB) where NB
   reduce(Base.FastMath.mul_fast, ntuple(Val(NB)) do i
      @inline 
      @inbounds BB[i][ϕ[i]]
   end)
end


@inline function BB_prod(ϕ::NTuple{NB}, BB, j) where NB
   reduce(Base.FastMath.mul_fast, ntuple(Val(NB)) do i
      @inline 
      @inbounds BB[i][j, ϕ[i]]
   end)
end


@inline function BB2_prod(ϕ::NTuple{NB}, BB1, BB2, j, z) where NB
   reduce(Base.FastMath.mul_fast, ntuple(Val(NB)) do i 
      if i == j 
         @inline 
         @inbounds BB2[i][ϕ[i]]
      elseif i == z
         @inline 
         @inbounds BB2[i][ϕ[i]]
      else
         @inline 
         @inbounds BB1[i][ϕ[i]]
      end
   end)
end

@inline function BB2_prod(ϕ::NTuple{NB}, BB1, BB2, j, m, n) where NB
   reduce(Base.FastMath.mul_fast, ntuple(Val(NB)) do i
      if i == m
         @inline 
         @inbounds BB2[i][j, ϕ[i]]
      elseif i == n
         @inline 
         @inbounds BB2[i][j, ϕ[i]]
      else
         @inline 
         @inbounds BB1[i][j, ϕ[i]]
      end
   end)
end

@inline function _prod_grad(b, ::Val{1})
   return (one(eltype(b)),)
end


@inline function _prod_grad(b::SVector{1, T}) where {T} 
   return b[1], SVector(one(T))
end


function _code_prod_grad(NB)
   code = Expr[] 
   # g[2] = b[1] 
   push!(code, :(g2 = b[1]))
   for i = 3:NB 
      # g[i] = g[i-1] * b[i-1]
      push!(code, Meta.parse("g$i = g$(i-1) * b[$(i-1)]"))
   end
   # h = b[N]
   push!(code, Meta.parse("h = b[$NB]"))
   for i = NB-1:-1:2
      # g[i] *= h
      push!(code, Meta.parse("g$i *= h"))
      # h *= b[i]
      push!(code, Meta.parse("h *= b[$i]"))
   end
   # g[1] = h
   push!(code, :(g1 = h))
   # return (g[1], g[2], ..., g[N])
   push!(code, Meta.parse(
            "return (" * join([ "g$i" for i = 1:NB ], ", ") * ")" ))
end

@inline @generated function _prod_grad(b, ::Val{NB}) where {NB}
   code = _code_prod_grad(NB)
   quote
      @fastmath begin 
         $(code...)
      end
   end
end

function _code_prod_grad_ed(NB)
   code = Expr[]
   # g[2] = b[1]
   push!(code, :(g2 = b[1]))
   for i = 3:NB
      # g[i] = g[i-1] * b[i-1]
      push!(code, Meta.parse("g$i = g$(i-1) * b[$(i-1)]"))
   end
   # h = b[N]
   push!(code, Meta.parse("h = b[$NB]"))
   for i = NB-1:-1:2
      # g[i] *= h
      push!(code, Meta.parse("g$i *= h"))
      # h *= b[i]
      push!(code, Meta.parse("h *= b[$i]"))
   end
   # g[1] = h
   push!(code, :(g1 = h))
   # return (g[1], g[2], ..., g[N])
   push!(code, :(g0 = g1 * b[1]))
   push!(code, Meta.parse(
            "return (" * join([ "g$i" for i = 0:NB ], ", ") * ")" ))
end

@inline @generated function _prod_grad_ed(b, ::Val{NB}) where {NB}
   code = _code_prod_grad_ed(NB)
   quote
      @fastmath begin 
         $(code...)
      end
   end
end
