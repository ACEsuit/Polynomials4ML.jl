

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
