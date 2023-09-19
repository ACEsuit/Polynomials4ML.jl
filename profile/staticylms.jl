
using Polynomials4ML, StaticArrays, BenchmarkTools, LinearAlgebra

# assume that rr = (x, y, z) is already normalized
function _strylm3(x, y, z)
   a0 = sqrt(3) 
   a1 = sqrt(3/4/π)
   a2 = sqrt(15 / 4 / π)
   a2 = sqrt(5) * a1 
   a20 = sqrt(5 / 16 / π)
   r² = x^2 + y^2 + z^2 


   return SA[ 
      # Y0 
      a0,
      # Y1 
      a1 * y,
      a1 * z,
      a1 * x,
      # Y2
      a2 * x*y,
      a2 * y*z,
      a20 * (3 * y^2 - 1), 
      a2 * x*z, 
      a2 * (x^2 - y^2) / 2,
      # Y3 
      sqrt(35/32/π) * y * (3*x^2 - y^2), 
      sqrt(105/4/π) * x*y*z, 
      sqrt(21/32/π) * y * (5 * z^2 - r²), 
      sqrt(7/16/π) * z * (5 * z^2 - r²),  
      sqrt(21/32/π) * x * (5 * z^2 - r²), 
      sqrt(105/4/π) * z * (x^2 - y^2), 
      sqrt(35/32/π) * x * (x^2 - 3 * y^2), 
      ]
end

##


function runn(N)
   for n = 1:N 
      x, y, z = rand(), rand(), rand()
      r = sqrt(x^2 + y^2 + z^2) 
      x = x/r; y = y/r; z = z/r   
      _strylm3(x, y, z)
   end
end

@btime runn(1000)


##

basis = RYlmBasis(3)
rr = @SVector randn(3) 
rr = rr / norm(rr)
Y = zeros(length(basis))
evaluate!(Y, basis, rr)

@btime evaluate!($Y, $basis, $rr)