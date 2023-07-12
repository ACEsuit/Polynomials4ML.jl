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

@inline function _prod_grad(b, ::Val{1})
   return (one(eltype(b)),)
end


# @inline function _prod_grad(b::SVector{1, T}) where {T} 
#    return b[1], SVector(one(T))
# end


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

function _pb_prod_grad(∂::Tuple, b::Tuple, vNB::Val{NB}) where {NB} 
   v∂ = SVector(∂...)
   vb = SVector(b...)
   g = _prod_grad(b, Val{NB}())
   ∇g = ForwardDiff.gradient(vb -> sum(∂ .* _prod_grad(vb.data, vNB)), vb).data 
   return g, ∇g 
end 


@inline function _prod_ed(b, ::Val{1})
   return (one(eltype(b)),)
end

@inline function _prod_ed(b::SVector{1, T}) where {T} 
   return b[1], SVector(one(T))
end

@inline function _prod_ed2(b, ::Val{1})
   return (one(eltype(b)),)
end

@inline function _prod_ed2(b::SVector{1, T}) where {T} 
   return b[1], SVector(one(T))
end

function _code_prod_ed(NB)
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

@inline @generated function _prod_ed(b, ::Val{NB}) where {NB}
   code = _code_prod_ed(NB)
   quote
      @fastmath begin 
         $(code...)
      end
   end
end

function _code_prod_ed2(NB)
   code = Expr[] 
   push!(code, Meta.parse("g$(2 * NB) = b[1]"))
   push!(code, Meta.parse("g$(NB + 2) = b[2]"))
   j = 2 * NB
   for i = 3:NB-1
       push!(code, Meta.parse("g$(2 * NB + i-2) = g$(2 * NB + i-3) * b[$i]"))
       m = j + NB - i + 1
       push!(code, Meta.parse("g$m = g$j * b[$(i - 1)]"))
       push!(code, Meta.parse("g$(NB+i) = g$(NB + i-1) * b[$i]"))
       for z = 1:NB - i - 1
          push!(code, Meta.parse("g$(m+z) = g$(m+z-1) * b[$(i+z)]"))
       end
       j = m
   end

   for i = 1:NB-1
      j = Int((i + 1) * NB - i/2 - i^2/2)
      push!(code, Meta.parse("g$i = g$j * b[$NB]"))
   end
   push!(code, Meta.parse("g$NB = g$j * b[$(NB-1)]"))
   # h = b[N]
   push!(code, Meta.parse("h = b[$NB]"))

   for i = NB-1:-1:3
      for z = 1:i-1
         # g[i] *= h
         j = Int(NB + i - 1 + (2 * NB - 2 - z) * (z-1)/2)
         push!(code, Meta.parse("g$j *= h"))
      end
      # h *= b[i]
      push!(code, Meta.parse("h *= b[$i]"))
   end
   
   # g[1] = h
   push!(code, Meta.parse("g$(NB + 1) = h"))
   push!(code, :(g0 = g1 * b[1]))
   # return (g[1], g[2], ..., g[N])
   push!(code, Meta.parse(
            "return (" * join([ "g$i" for i = 0:Int(NB+NB*(NB-1)/2) ], ", ") * ")" ))
end

@inline @generated function _prod_ed2(b, ::Val{NB}) where {NB}
   code = _code_prod_ed2(NB)
   quote
      @fastmath begin 
         $(code...)
      end
   end
end

