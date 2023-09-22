using BenchmarkTools, LoopVectorization

sizeY(maxL) = (maxL + 1) * (maxL + 1)
index_y(l, m) = m + l + (l*l) + 1
sizeP(maxL) = div((maxL + 1) * (maxL + 2), 2)
index_p(l, m) = m + div(l*(l+1), 2) + 1


function _cYlm!(Y, L, cosθ, sinθ, P, t)
	nS = length(cosθ)
	@assert size(P) == (nS, sizeP(L))
	@assert size(Y) == (nS, sizeY(L))
   @assert length(sinθ) == length(t) == nS 

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


function _rYlm!(Yr, Yi, L, cosθ, sinθ, P, tr, ti)
	nS = length(cosθ)
	@assert size(P) == (nS, sizeP(L))
	@assert size(Yr) == size(Yi) == (nS, sizeY(L))
   @assert length(sinθ) == length(tr) == length(ti) == nS 

   fill!(t, 1 / sqrt(2) + im * 0)

	@inbounds begin 
		for l = 0:L 
			i_yl0 = index_y(l, 0)
			i_pl0 = index_p(l, 0)
			@avx for i = 1:nS
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
					Yr[i, i_ylm⁻] = (sig * p) * tr[i] # conj(t[i])
					Yi[i, i_ylm⁻] = - (sig * p) * ti[i] # conj(t[i])
					Yr[i, i_ylm⁺] = tr[i] * p  
					Yi[i, i_ylm⁺] = ti[i] * p  
				end
			end
		end
	end 

	return Y
end

##

L = 8
nX = 1024 
cosθ = [ cos(π * rand()) for _=1:nX ]   # input 
sinθ = [ sin(π * rand()) for _=1:nX ] 
t = zeros(ComplexF64, nX)   # temporary array 
tr = real.(t) 
ti = imag.(t)
P = rand(nX, sizeP(L)) # in the real code, these are the ALPs
Y = zeros(ComplexF64, nX, sizeY(L))   # allocate output
Yr = real.(Y)
Yi = imag.(Y)


##

@btime _cYlm!($Y, $L, $cosθ, $sinθ, $P, $t)
# 37.541 μs (0 allocations: 0 bytes)
  
@btime _rYlm!($Yr, $Yi, $L, $cosθ, $sinθ, $P, $tr, $ti)
# 31.042 μs (0 allocations: 0 bytes)
#   ~ ca 54 us without avx. 

##

# with view into the original array 
Yre = reinterpret(Float64, Y)
Yre_r = @view Yre[1:2:end, :]
Yre_i = @view Yre[2:2:end, :]

@btime _rYlm!($Yre_r, $Yre_i, $L, $cosθ, $sinθ, $P, $tr, $ti)
# 43.333 μs (0 allocations: 0 bytes)


## 

# @profview let Y= Y, L=L, cosθ=cosθ, sinθ=sinθ, P=P, t =t 
#    for n = 1:50_000 
#       _cYlm!(Y, L, cosθ, sinθ, P, t)
#    end
# end

# @profview let Yr = Yr, Yi = Yi, L = L, cosθ = cosθ, sinθ = sinθ, P = P, tr = tr, ti = ti
#    for n = 1:66_000 
#       _rYlm!(Yr, Yi, L, cosθ, sinθ, P, tr, ti)
#    end
# end

