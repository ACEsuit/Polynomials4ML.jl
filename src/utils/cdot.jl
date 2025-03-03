

_cdot(a::Real, b::Real) = a * b 

_cdot(a::Real, b::Complex) = a * real(b)

_cdot(a::Complex, b::Real) = real(a) * b 

_cdot(a::Complex, b::Complex) = 
      real(a) * real(b) + imag(a) * imag(b)

_cdot(a::SVector, b::Number) = _cdot.(a, Ref(b))
_cdot(a::Number, b::SVector) = _cdot.(a, Ref(b))

