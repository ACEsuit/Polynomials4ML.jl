
using Polynomials4ML, BenchmarkTools
using Polynomials4ML.Testing: time_standard!, time_batched!
##

L = 8; nX = 128
rYlm = basis = RYlmBasis(L)
cYlm = CYlmBasis(L)
X = [Polynomials4ML.rand_sphere() for _=1:nX]
P_batched = evaluate(basis, X)
P_standard = copy(P_batched')

time_standard!(P_standard, basis, X)
time_batched!(P_batched, basis, X)

##

@info("evaluate!")
@info("Naive Implementation")
@btime time_standard!($P_standard, $basis, $X)
@info("Batched Implementation")
@btime time_batched!($P_batched, $basis, $X)

## 

@info("Real vs Complex Ylm")

@info("Batched Implementation")
cP_batched = evaluate(cYlm, X)
@btime time_batched!($P_batched, $rYlm, $X)
@btime time_batched!($cP_batched, $cYlm, $X)

##


# @profview let basis = basis, X = X, P_batched = P_batched 
#    for n = 1:100_000 
#       time_batched!(P_batched, basis, X)
#    end
# end

##

# @profview let alp = alp, S = S, P_batched = P_batched 
#    for n = 1:100_000 
#       time_batched!(P_batched, alp, S)
#    end
# end


##

using LoopVectorization

sizeY(maxL) = (maxL + 1) * (maxL + 1)
index_y(l::Integer, m::Integer) = m + l + (l*l) + 1
sizeP(maxL) = div((maxL + 1) * (maxL + 2), 2)
index_p(l::Integer,m::Integer) = m + div(l*(l+1), 2) + 1

function rYlm!(Y, L, P::AbstractMatrix, cosφ, sinφ, cosmφ, sinmφ)
   nX = length(cosφ)
	@assert nX == length(sinφ) == length(cosmφ) == length(sinmφ) == size(Y, 1) == size(P, 1)
   @assert size(P, 2) >= sizeP(L)
	@assert size(Y, 2) >= sizeY(L)
   
   @inbounds begin 
		fill!(sinmφ, 0)
		fill!(cosmφ, 1)

		oort2 = 1 / sqrt(2)
		for l = 0:L
			i_yl0 = index_y(l, 0); i_pl0 = index_p(l, 0)
			@avx for i = 1:nX
				Y[i, i_yl0] = P[i, i_pl0] * oort2
			end
		end

		for m in 1:L
			@avx for i = 1:nX
				cmi, smi = cosmφ[i], sinmφ[i]
				cosmφ[i] = cmi * cosφ[i] - smi * sinφ[i]
				sinmφ[i] = smi * cosφ[i] + cmi * sinφ[i]
			end

			for l in m:L
				i_plm = index_p(l, m)
				i_ylm⁺ = index_y(l, m)
				i_ylm⁻ = index_y(l, -m)
				@avx for i = 1:nX
					p = P[i, i_plm]
					Y[i, i_ylm⁺] =  p * cosmφ[i]
					Y[i, i_ylm⁻] = -p * sinmφ[i]
				end
			end
		end

   end # inbounds 
	return Y
end

##

nX = 128 
L = 8 
cosφ = rand(nX); sinφ = sqrt.(1 .- cosφ.^2)
cosmφ = similar(cosφ); sinmφ = similar(sinφ)
Y = zeros(nX, sizeY(L))
P = rand(nX, sizeP(L))

@btime rYlm!($Y, $L, $P, $cosφ, $sinφ, $cosmφ, $sinmφ)


##

@profview let Y = Y, L = L, P = P, cosφ = cosφ, sinφ = sinφ, cosmφ = cosmφ, sinmφ = sinmφ
   for n = 1:3_000_000 
      rYlm!(Y, L, P, cosφ, sinφ, cosmφ, sinmφ)
   end
end
