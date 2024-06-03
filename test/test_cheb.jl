using Polynomials4ML, Test
using Polynomials4ML: evaluate, evaluate_d, evaluate_dd
using Polynomials4ML.Testing: println_slim, print_tf, test_derivatives, 
                              test_withalloc 


##

@info("Testing Real Chebyshev Polynomials (ChebBasis)")
N = 10
basis = ChebBasis(N) 
basis2 = chebyshev_basis(N; normalize=false)

@info("      correctness")
mm = natural_indices(basis)
print_tf(@test mm == 0:N-1)

θ = 2*π * rand()
x = cos(θ)
r = basis(x) ./ basis2(x)
for ntest = 1:30
   local θ
   local x
   θ = 2*π * rand()
   x = cos(θ)
   P = basis(x)
   P2 = [ cos(m*θ) for m in mm ]
   P3 = basis2(x)
   print_tf(@test P ≈ P2 && (P ./ P3 ≈ r))
end
println() 

##

@info("      test derivatives")
generate_x = () -> 2*rand()-1
test_derivatives(basis, generate_x)

##

using Bumper, WithAlloc, BenchmarkTools, StaticArrays
using ObjectPools: unwrap
using WithAlloc: withalloc
using Polynomials4ML: evaluate!, evaluate_d!, evaluate_ed!

@info("        test Bumper usage") 
let basis = basis 
   x = generate_x()
   xx = [ generate_x() for _ = 1:32 ]

   @no_escape begin 
      P1 = basis(x) 
      P2 = @withalloc evaluate!(basis, x)
      println_slim( @test unwrap(P1) ≈ P2 )
      P1b = basis(xx) 
      P2b = @withalloc evaluate!(basis, xx)
      println_slim( @test unwrap(P1b) ≈ P2b )
   end
end

##

@info("        test withalloc")
println_slim(@test test_withalloc(basis, 0.5)  )
println_slim(@test test_withalloc(basis, [ generate_x() for _ = 1:16 ]) )
println_slim(@test test_withalloc(basis2, 0.5)  )
println_slim(@test test_withalloc(basis2, [ generate_x() for _ = 1:16 ]) )
