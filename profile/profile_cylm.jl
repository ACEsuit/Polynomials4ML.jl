
using Polynomials4ML, BenchmarkTools
using Polynomials4ML.Testing: time_standard!, time_batched!
##

L = 8; nX = 1024
basis = CYlmBasis(L)
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

# @profview let basis = basis, X = X, P_batched = P_batched 
#    for n = 1:30_000 
#       time_batched!(P_batched, basis, X)
#    end
# end

##

alp = basis.alp
S = Polynomials4ML.cart2spher.(X)

P_batched = evaluate(alp, S)
P_standard = copy(P_batched')

time_standard!(P_standard, alp, S)
time_batched!(P_batched, alp, S) 

##
@info("evaluate!")
@info("Naive Implementation")
@btime time_standard!($P_standard, $alp, $S)
@info("Batched Implementation")
@btime time_batched!($P_batched, $alp, $S)

##

# @profview let alp = alp, S = S, P_batched = P_batched 
#    for n = 1:100_000 
#       time_batched!(P_batched, alp, S)
#    end
# end

##

@btime Polynomials4ML.evaluate!($P_batched, $alp, $S)


##

using Polynomials4ML: sizeP, sizeY, index_p, index_y
using LoopVectorization

function _cYlm!(Y, L, cosθ, sinθ, P, t)
	nS = length(cosθ) 
	@assert size(P, 1) >= nS 
	@assert size(Y, 1) >= nS 
   @assert size(P, 2) >= sizeP(L)
	@assert size(Y, 2) >= sizeY(L)
   @assert length(sinθ) == nS 
   @assert length(t) >= nS 

   fill!(t, 1 / sqrt(2) + im * 0)

	@inbounds begin 
		for l = 0:L 
			i_yl0 = index_y(l, 0)
			i_pl0 = index_p(l, 0)
			for i = 1:nS
				Y[i, i_yl0] = P[i, i_pl0] * t[i] 
			end
		end

		sig = 1
		for m in 1:L
			sig *= -1
		   for i = 1:nS
				t[i] *= cosθ[i] + im * sinθ[i]
			end

		   for l in m:L
				i_plm = index_p(l,m)
				i_ylm⁺ = index_y(l,  m)
				i_ylm⁻ = index_y(l, -m)
				for i = 1:nS
					p = P[i, i_plm]
					Y[i, i_ylm⁻] = (sig * p) * conj(t[i])
					Y[i, i_ylm⁺] = t[i] * p  
				end
			end
		end
	end 

	return Y
end

function _cYlm!(Yr, Yi, L, cosθ, sinθ, P, tr, ti)
	nS = length(cosθ) 
	@assert size(P, 1) >= nS 
	@assert size(Yr, 1) >= nS 
	@assert size(Yi, 1) >= nS 
   @assert size(P, 2) >= sizeP(L)
	@assert size(Yr, 2) >= sizeY(L)
	@assert size(Yi, 2) >= sizeY(L)
   @assert length(sinθ) == nS 
   @assert length(tr) >= nS 
   @assert length(ti) >= nS 

   fill!(tr, 1 / sqrt(2) )
   fill!(ti, 0)

	@inbounds begin 
		for l = 0:L 
			i_yl0 = index_y(l, 0)
			i_pl0 = index_p(l, 0)
			for i = 1:nS
				Yr[i, i_yl0] = P[i, i_pl0] * tr[i] 
				Yi[i, i_yl0] = P[i, i_pl0] * ti[i] 
			end
		end

		sig = 1
		for m in 1:L
			sig *= -1
		   @avx for i = 1:nS
				tr[i] = tr[i] * cosθ[i] - ti[i] * sinθ[i]
            ti[i] = ti[i] * cosθ[i] + tr[i] * sinθ[i]
			end

		   for l in m:L
				i_plm = index_p(l,m)
				i_ylm⁺ = index_y(l,  m)
				i_ylm⁻ = index_y(l, -m)
				@avx for i = 1:nS
					p = P[i, i_plm]
					# Y[i, i_ylm⁻] = (sig * p) * conj(t[i])
					# Y[i, i_ylm⁺] = t[i] * p  
					Yr[i, i_ylm⁻] = sig * p * ti[i]
					Yr[i, i_ylm⁺] = p * tr[i]
					Yi[i, i_ylm⁻] = sig * p * tr[i]
					Yi[i, i_ylm⁺] = ti[i] * p  
				end
			end
		end
	end 

	return  nothing 
end

##

L = basis.alp.L
cosθ = [ s.cosθ for s in S ]
sinθ = [ s.sinθ for s in S ]
t = zeros(ComplexF64, nX)
P = rand(nX, sizeP(L))
Y = zeros(ComplexF64, nX, sizeY(L))
tr = real.(t) 
ti = imag.(t) 
Yr = real.(Y) 
Yi = imag.(Y) 

##

@btime _cYlm!($Y, $L, $cosθ, $sinθ, $P, $t)

@btime _cYlm!($Yr, $Yi, $L, $cosθ, $sinθ, $P, $tr, $ti)

## 



# @profview let Y= Y, L=L, cosθ=cosθ, sinθ=sinθ, P=P, t =t 
#    for n = 1:50_000 
#       _cYlm!(Y, L, cosθ, sinθ, P, t)
#    end
# end
