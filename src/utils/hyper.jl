
using HyperDualNumbers: Hyper

## ---------- HyperDualNumbers utils ---------
Base.atan(y::Hyper{T}, x::Hyper{T}) where {T} = 
      atan(y/x)*(x != 0) + (1-2*(y<0))*(pi*(x<0) + 1/2*pi*(x==0))
