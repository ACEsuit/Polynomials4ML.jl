module Testing

using Polynomials4ML: evaluate!, evaluate_ed!, evaluate_ed2!, 
               evaluate, evaluate_d, evaluate_ed, evaluate_dd, evaluate_ed2 , 
               _alloc

using Test, ForwardDiff

using ACEbase.Testing: print_tf, println_slim



function time_standard!(P, basis, X)
   for i = 1:length(X)
      Pi = @view P[:, i]
      evaluate!( Pi, basis, X[i] )
   end
   return P 
end

time_batched!(P, basis, X) = evaluate!(P, basis, X)

function time_ed_standard!(P, dP, basis, X)
   for i = 1:length(X)
      evaluate_ed!( (@view P[:, i]), (@view dP[:, i]), basis, X[i] )
   end
   return P, dP  
end

time_ed_batched!(P, dP, basis, X) = evaluate_ed!(P, dP, basis, X)

function time_ed2_standard!(P, dP, ddP, basis, X)
   for i = 1:length(X)
      evaluate_ed2!( (@view P[:, i]), (@view dP[:, i]), (@view ddP[:, i]), basis, X[i] )
   end
   return P, dP  
end

time_ed2_batched!(P, dP, ddP, basis, X) = evaluate_ed2!(P, dP, ddP, basis, X)



# ------------------------ Test correctness of derivatives 


function test_derivatives(basis, generate_x, nX = 32, ntest = 8)

   @info("Test consistency of evaluate_**")
   for ntest = 1:ntest 
      x = generate_x()
      P1 = evaluate(basis, x)
      P2, dP2 = evaluate_ed(basis, x)
      dP3 = evaluate_d(basis, x)
      P4, dP4, ddP4 = evaluate_ed2(basis, x)
      ddP5 = evaluate_dd(basis, x)
      print_tf(@test P1 ≈ P2 ≈ P4 )
      print_tf(@test dP2 ≈ dP3 ≈ dP4)
      print_tf(@test ddP4 ≈ ddP5)
   end
   println() 

   @info("Test correctness of derivatives")
   for ntest = 1:ntest 
      x = generate_x()
      P, dP, ddP = evaluate_ed2(basis, x)

      adP = ForwardDiff.derivative(x -> evaluate(basis, x), x)
      print_tf(@test adP ≈ dP)
      
      addP = ForwardDiff.derivative(x -> evaluate_d(basis, x), x)
      print_tf(@test addP ≈ ddP)
   end 
   println() 

   @info("Test consistency of batched evaluation")

   X = [ generate_x() for _ = 1:nX ]
   bP1 = _alloc(basis, X)
   bdP1 = _alloc(basis, X)
   bddP1 = _alloc(basis, X)
   for (i, x) in enumerate(X)
      bP1[i, :] = evaluate(basis, x)
      bdP1[i, :] = evaluate_d(basis, x)
      bddP1[i, :] = evaluate_dd(basis, x)
   end
      
   bP2 = evaluate(basis, X)
   bP3, bdP3 = evaluate_ed(basis, X)
   bP4, bdP4, bddP4 = evaluate_ed2(basis, X)
      
   println_slim(@test bP2 ≈ bP1 ≈ bP3 ≈ bP4)
   println_slim(@test bdP3 ≈ bdP1 ≈ bdP4)
   println_slim(@test bddP4 ≈ bddP1)
   
end


end