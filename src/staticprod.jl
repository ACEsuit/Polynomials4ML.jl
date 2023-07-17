

_static_prod(b::NTuple{N, T}) where {N, T <: Number} = 
          Base.FastMath.mul_fast(b[1], _static_prod(b[2:N]))

_static_prod(b::NTuple{1, T}) where {T <: Number} = b[1]

function _static_prod_ed(b::NTuple{N, T}) where {N, T <: Number} 
   b2 = b[2:N]
   p2, g2 = _static_prod_ed(b2)
   return b[1] * p2, tuple(p2, ntuple(i -> b[1] * g2[i], N-1)...)
end

function _static_prod_ed(b::NTuple{1, T}) where {N, T <: Number} 
   return b[1], (one(T),)
end


function _static_prod_ed2(b::NTuple{N, T}) where {N, T <: Number}
   b2 = b[2:N]
   p2, g2, h22 = _static_prod_ed2(b2)
   p = b[1] * p2
   g = tuple(p2, ntuple(i -> b[1] * g2[i], N-1)...) 
   h1 = SVector(g2...)
   h22_ = b[1] * h22 
   h = vcat( SMatrix{1,N}(zero(T), h1...), hcat(h1, h22_))
   return p, g, h 
end

function _static_prod_ed2(b::NTuple{1, T}) where {N, T <: Number} 
   return b[1], (one(T),), SMatrix{1,1}(zero(T))
end


function _pb_grad_static_prod(∂::NTuple{N, T}, b::NTuple{N, T}) where {N, T <: Number} 
   ∂2 = ∂[2:N]
   b2 = b[2:N]
   p2, g2, u2 = _pb_grad_static_prod(∂2, b2)
   return b[1] * p2, 
          tuple(p2, ntuple(i -> b[1] * g2[i], N-1)...), 
          tuple(sum(∂2 .* g2), ntuple(i -> ∂[1] * g2[i] + b[1] * u2[i], N-1)...)
end

function _pb_grad_static_prod(∂::NTuple{1, T}, b::NTuple{1, T}) where {N, T <: Number} 
   return b[1], (one(T),), (zero(T),)
end


## TODO: work out how to recursively evaluate ed2
# @inline function _prod_ed2(b, ::Val{1})
#    return (one(eltype(b)),)
# end

# @inline function _prod_ed2(b::SVector{1, T}) where {T} 
#    return b[1], SVector(one(T))
# end

# function _code_prod_ed2(NB)
#    code = Expr[] 
#    push!(code, Meta.parse("g$(2 * NB) = b[1]"))
#    push!(code, Meta.parse("g$(NB + 2) = b[2]"))
#    j = 2 * NB
#    for i = 3:NB-1
#        push!(code, Meta.parse("g$(2 * NB + i-2) = g$(2 * NB + i-3) * b[$i]"))
#        m = j + NB - i + 1
#        push!(code, Meta.parse("g$m = g$j * b[$(i - 1)]"))
#        push!(code, Meta.parse("g$(NB+i) = g$(NB + i-1) * b[$i]"))
#        for z = 1:NB - i - 1
#           push!(code, Meta.parse("g$(m+z) = g$(m+z-1) * b[$(i+z)]"))
#        end
#        j = m
#    end

#    for i = 1:NB-1
#       j = Int((i + 1) * NB - i/2 - i^2/2)
#       push!(code, Meta.parse("g$i = g$j * b[$NB]"))
#    end
#    push!(code, Meta.parse("g$NB = g$j * b[$(NB-1)]"))
#    # h = b[N]
#    push!(code, Meta.parse("h = b[$NB]"))

#    for i = NB-1:-1:3
#       for z = 1:i-1
#          # g[i] *= h
#          j = Int(NB + i - 1 + (2 * NB - 2 - z) * (z-1)/2)
#          push!(code, Meta.parse("g$j *= h"))
#       end
#       # h *= b[i]
#       push!(code, Meta.parse("h *= b[$i]"))
#    end
   
#    # g[1] = h
#    push!(code, Meta.parse("g$(NB + 1) = h"))
#    push!(code, :(g0 = g1 * b[1]))
#    # return (g[1], g[2], ..., g[N])
#    push!(code, Meta.parse(
#             "return (" * join([ "g$i" for i = 0:Int(NB+NB*(NB-1)/2) ], ", ") * ")" ))
# end

# @inline @generated function _prod_ed2(b, ::Val{NB}) where {NB}
#    code = _code_prod_ed2(NB)
#    quote
#       @fastmath begin 
#          $(code...)
#       end
#    end
# end

