using Test
using Polynomials4ML.Testing: println_slim, print_tf
using Printf
using Polynomials4ML: SparseProduct, evaluate, evaluate_ed, evaluate_ed2
using LinearAlgebra: norm
using Polynomials4ML
using ACEbase.Testing: fdtest
using Zygote

test_evaluate(basis::SparseProduct, BB::Tuple{Vararg{<: AbstractVector}}) = 
       [ prod(BB[j][basis.spec[i][j]] for j = 1:length(BB)) 
            for i = 1:length(basis) ]

# test_evaluate(basis::SparseProduct, BB::Tuple{Vararg{<: AbstractMatrix}}) = 
#         [ test_evaluate(basis, ntuple(i -> BB[i][j, :], length(BB)))
#          for j = 1:size(BB[1], 1) )            


##
NB = 3 # For _rrule_evaluate test we need NB = 3, fix later by generalizing the test case


N = [i * 4 for i = 1:NB]

B = [randn(N[i]) for i = 1:NB]

spec = sort([ Tuple([rand(1:N[i]) for i = 1:NB]) for _ = 1:6])

basis = SparseProduct(spec)

test_evaluate(basis::SparseProduct, BB::Tuple) = 
       [ prod(BB[j][basis.spec[i][j]] for j = 1:length(BB)) 
            for i = 1:length(basis) ]

function test_evaluate_ed(basis, BB)
    A = deepcopy(evaluate_ed(basis, BB)[1])
    dA = evaluate_ed(basis, BB)[2]
    errors = Float64[]
    # loop through finite-difference step-lengths
    @printf("---------|----------- \n")
    @printf("    h    | error \n")
    @printf("---------|----------- \n")
    for p = 2:11
        h = 0.1^p
        Δ = []
        for n = 1:length(dA) # basis
            for i = 1:length(dA[n]) #NB
                for j = 1:length(dA[n][i]) #BB[i]
                    BB[i][j] += h
                    push!(Δ, dA[n][i][j] - (evaluate(basis, BB)[n] - A[n])/h)
                    BB[i][j] -= h
                end
            end
        end
        push!(errors, norm(Δ, Inf))
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
       Δ = []
       for n = 1:length(ddA) # basis
           for i = 1:length(ddA[n]) #NB
               for j = 1:length(ddA[n][i]) #BB[i]
                   BB[i][j] += h
                   AA = evaluate(basis, BB)[n] - 2 * A[n]
                   BB[i][j] -= 2*h
                   AA = (AA + evaluate(basis, BB)[n])/h^2
                   BB[i][j] += h 
                   push!(Δ, ddA[n][i][j] - AA)
               end
           end
       end
       push!(errors, norm(Δ, Inf))
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

@info("Test serial evaluation")

BB = Tuple(B)

A1 = test_evaluate(basis, BB)
A2 = evaluate(basis, BB)

println_slim(@test A1 ≈ A2 )

@info("Test serial evaluation_ed")
BB = Tuple(B)

test_evaluate_ed(basis, BB)

AA = evaluate(basis, BB)
A1 = evaluate_ed(basis, BB)[1]

println_slim(@test AA ≈ A1 )
##

@info("Test serial evaluation_ed2")
BB = Tuple(B)

test_evaluate_ed2(basis, BB)

AA = evaluate(basis, BB)
dA = evaluate_ed(basis, BB)[2]
A1 = evaluate_ed2(basis, BB)[1]
A2 = evaluate_ed2(basis, BB)[2]

println_slim(@test AA ≈ A1 )
Δ = maximum([norm(dA[i][j] - A2[i][j], Inf) for i = 1:length(dA) for j = 1:length(dA[i])])
println_slim(@test norm(Δ) <= 1e-15)
@info("Test batch evaluation")

nX = 5
bBB = Tuple([randn(nX, N[i]) for i = 1:NB])
bA1 = zeros(ComplexF64, nX, length(basis))

for j = 1:nX
    bA1[j, :] = evaluate(basis, Tuple([bBB[i][j, :] for i = 1:NB]))
end

bA2 = evaluate(basis, bBB)

println_slim(@test bA1 ≈ bA2)

@info("Test batch evaluate_ed")

nX = 2
bBB = Tuple([randn(nX, N[i]) for i = 1:NB])
A1 = zeros(ComplexF64, nX, length(basis))
_similar(BB::Tuple) = Tuple([similar(BB[i]) for i = 1:length(BB)])
bA1 = [_similar(bBB) for j = 1:length(basis)]  # nX * basis

for j = 1:nX
    A1[j, :] = evaluate_ed(basis, Tuple([bBB[i][j, :] for i = 1:NB]))[1]
end
for i = 1:length(basis)
    for j = 1:NB
        for z = 1:nX
            bA1[i][j][z,:] = (evaluate_ed(basis, Tuple([bBB[i][z, :] for i = 1:NB]))[2][i][j])
        end
    end
end

A2 = evaluate_ed(basis, bBB)[1]
bA2 = evaluate_ed(basis, bBB)[2]

println_slim(@test A1 ≈ A2)

Δ = maximum([norm(bA1[i][j] - bA2[i][j], Inf) for i = 1:length(bA1) for j = 1:length(bA1[i])])
println_slim(@test Δ ≈ 0)
## 

@info("Test batch evaluate_ed2")

nX = 64 
bBB = Tuple([randn(nX, N[i]) for i = 1:NB])
A1 = zeros(ComplexF64, nX, length(basis))
_similar(BB::Tuple) = Tuple([similar(BB[i]) for i = 1:length(BB)])
bA1 = [_similar(bBB) for j = 1:length(basis)]  # nX * basis
bbA1 = [_similar(bBB) for j = 1:length(basis)] 

for j = 1:nX
    A1[j, :] = evaluate_ed2(basis, Tuple([bBB[i][j, :] for i = 1:NB]))[1]
end
for i = 1:length(basis)
    for j = 1:NB
        for z = 1:nX
            bA1[i][j][z,:] = (evaluate_ed2(basis, Tuple([bBB[i][z, :] for i = 1:NB]))[2][i][j])
            bbA1[i][j][z,:] = (evaluate_ed2(basis, Tuple([bBB[i][z, :] for i = 1:NB]))[3][i][j])
        end
    end
end

A2 = evaluate_ed2(basis, bBB)[1]
bA2 = evaluate_ed2(basis, bBB)[2]
bbA2 = evaluate_ed2(basis, bBB)[3]

println_slim(@test A1 ≈ A2)
Δ = maximum([norm(bA1[i][j] - bA2[i][j], Inf) for i = 1:length(bA1) for j = 1:length(bA1[i])])
println_slim(@test norm(Δ) <= 1e-15)
Δ = maximum([norm(bbA1[i][j] - bbA2[i][j], Inf) for i = 1:length(bbA1) for j = 1:length(bbA1[i])])
println_slim(@test norm(Δ) <= 1e-15)

@info("Testing _rrule_evaluate")
using LinearAlgebra: dot 

N1 = 10
N2 = 20
N3 = 30

for ntest = 1:30
    local bBB
    local bUU
    local bA2
    bBB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
    bUU = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
    _BB(t) = ( bBB[1] + t * bUU[1], bBB[2] + t * bUU[2], bBB[3] + t * bUU[3] )
    bA2 = Polynomials4ML.evaluate(basis, bBB)
    u = randn(size(bA2))
    F(t) = dot(u, Polynomials4ML.evaluate(basis, _BB(t)))
    dF(t) = begin
        val, pb = Zygote.pullback(evaluate, basis, _BB(t))
        ∂BB = pb(u)[2] # pb(u)[1] returns NoTangent() for basis argument
        return sum( dot(∂BB[i], bUU[i]) for i = 1:length(bUU) )
    end
    print_tf(@test fdtest(F, dF, 0.0; verbose=false))
end
println()

##

# try with rrule
u, pb = Zygote.pullback(evaluate, basis, bBB)
# u1, pb1 = Polynomials4ML._rrule_evaluate(basis, bBB)


# TODO: look into why this is failing
# using ChainRulesTestUtils
# test_rrule(evaluate, basis, bBB)
