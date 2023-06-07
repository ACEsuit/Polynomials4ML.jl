
#
# this script explores HyperDualNumbers to implement the 
# laplacian operator 
#

using Polynomials4ML, HyperDualNumbers, Lux, LuxCore, Random
P4ML = Polynomials4ML
rng = Random.default_rng()

##

Pn = legendre_basis(10)
l_Pn = P4ML.lux(Pn)

l_embed = BranchLayer(;Pn = l_Pn,)

bA = P4ML.PooledSparseProduct([ (n,) for n = 1:length(Pn) ])
l_bA = P4ML.lux(bA)

ch1 = Chain(embed = l_embed, A = l_bA)

##

xx = 2*rand(5) .- 1

ps, st = Lux.setup(rng, ch1)
o, _ = ch1(xx, ps, st)

P = Pn(xx)
A = bA( (P,) )

## 

hxx = [ Hyper(xx[i], i==1, i==1, 0) for i = 1:length(xx) ] 
hA1 = bA( (Pn(hxx),) )

hA2, _ = ch1(hxx, ps, st)
hA1 == hA2

##

using BenchmarkTools

@btime $bA( ($Pn( $xx ),) )
@btime $ch1( $xx, $ps, $st )
@btime $bA( ($Pn( $hxx ),) )
@btime $ch1( $hxx, $ps, $st )


##

# add another layer to the chain - n-correlations

spec = [ [ [n1,] for n1 = 1:length(Pn) ]; 
         [ [n1, n2] for n1 = 1:length(Pn) for n2 = n1:length(Pn) ] ]
bAA = P4ML.SparseSymmProd(spec)
l_bAA = P4ML.lux(bAA)

ch2 = Chain(embed = l_embed, A = l_bA, AA = l_bAA)
ps, st = Lux.setup(rng, ch2)

ch2(xx, ps, st)
ch2(hxx, ps, st)


## 
# Most important test: make a model, take a gradient, then run the 
# Hypers through the gradient ... 

module M1
   using LuxCore, LinearAlgebra, Random 
   import LuxCore:  AbstractExplicitLayer, initialparameters, initialstates
   struct DotL <: AbstractExplicitLayer
      nin::Int
   end
   function (l::DotL)(x::AbstractVector{<: Number}, ps, st)
      return dot(x, ps.W), st
   end
   initialparameters(rng::AbstractRNG, l::DotL) = ( W = randn(rng, l.nin), )
   initialstates(rng::AbstractRNG, l::DotL) = NamedTuple()
end

ch3 = Chain(embed = l_embed, A = l_bA, AA = l_bAA, dot = M1.DotL(length(bAA)))
ps, st = Lux.setup(rng, ch3)

ch3(xx, ps, st)
ch3(hxx, ps, st)


## 

using Zygote

g_ch3 = xx -> Zygote.gradient(p -> ch3(xx, p, st)[1], ps)[1]
g_ch3(xx)
g_ch3(hxx)


##

module NTarrays
   # using NamedTupleTools

   struct NTarr{NTT}
      nt::NTT
   end

   export array

   array(nt::NamedTuple) = NTarr(nt)

   # ------------------------------
   #  0 

   zero!(a::AbstractArray) = fill!(a, zero(eltype(a)))
   zero!(a::Nothing) = nothing 

   function zero!(nt::NamedTuple)
      for k in keys(nt)
         zero!(nt[k])
      end
      return nt
   end 

   Base.zero(nt::NamedTuple) = zero!(deepcopy(nt))

   Base.zero(nt::NTarr) = NTarr(zero(nt.nt))

   # ------------------------------
   #  + 


   function _add!(a1::AbstractArray, a2::AbstractArray) 
      a1[:] .= a1[:] .+ a2[:]
      return nothing 
   end

   _add!(at::Nothing, args...) = nothing 

   function _add!(nt1::NamedTuple, nt2)
      for k in keys(nt1)
         _add!(nt1[k], nt2[k])
      end
      return nothing 
   end

   function _add(nt1::NamedTuple, nt2::NamedTuple)
      nt = deepcopy(nt1)
      _add!(nt, nt2)
      return nt
   end

   Base.:+(nt1::NTarr, nt2::NTarr) = NTarr(_add(nt1.nt, nt2.nt))

   # ------------------------------
   #  * 

   _mul!(::Nothing, args... ) = nothing 

   function _mul!(a::AbstractArray, λ::Number)
      a[:] .= a[:] .* λ
      return nothing 
   end

   function _mul!(nt::NamedTuple, λ::Number)
      for k in keys(nt)
         _mul!(nt[k], λ)
      end
      return nothing 
   end

   function _mul(nt::NamedTuple, λ::Number)
      nt = deepcopy(nt)
      _mul!(nt, λ)
      return nt
   end

   Base.:*(λ::Number, nt::NTarr) = NTarr(_mul(nt.nt, λ))
   Base.:*(nt::NTarr, λ::Number) = NTarr(_mul(nt.nt, λ))

   # ------------------------------
   #   map 

   _map!(f, a::AbstractArray) = map!(f, a, a) 

   _map!(f, ::Nothing) = nothing 

   function _map!(f, nt::NamedTuple)
      for k in keys(nt)
         _map!(f, nt[k])
      end
      return nothing 
   end

   function Base.map!(f, dest::NTarr, src::NTarr)
      _map!(f, nt.nt)
      return nt
   end

end 

using Main.NTarrays

function laplacian(gfun, xx)
   function _mapadd!(f, dest::NamedTuple, src::NamedTuple) 
      for k in keys(dest)
         _mapadd!(f, dest[k], src[k])
      end
      return nothing 
   end
   _mapadd!(f, dest::Nothing, src) = nothing
   _mapadd!(f, dest::AbstractArray, src::AbstractArray) = 
            map!((s, d) -> d + f(s), dest, src, dest)

   Δ = NTarrays.zero!(gfun(xx))
   for i = 1:length(xx) 
      hxx = [ Hyper(xx[j], j==i, j==i, 0) for j = 1:length(xx) ]
      _mapadd!(ε₁ε₂part, Δ, gfun(hxx))
   end
   return Δ
end

Δ1 = laplacian(g_ch3, xx)

@btime $g_ch3($xx)
@btime $laplacian($g_ch3, $xx)
# test the correctness of the implementation 

using LinearAlgebra

function laplacian_fd(gfun, xx; h = 1e-4)
   Nx = length(xx)
   g0 = array(gfun(xx))
   Δ = g0 * (-2 * Nx)
   for i = 1:Nx
      xxp = [ xx[j] + h * (i==j) for j = 1:Nx ]
      xxm = [ xx[j] - h * (i==j) for j = 1:Nx ]      
      Δ = Δ + (array(gfun(xxp)) + array(gfun(xxm)))
   end
   return (Δ * (1/h^2)).nt
end

for h in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
   Δ2 = laplacian_fd(g_ch3, xx, h = h)
   println("h = $h, error = $(norm(Δ1.dot.W - Δ2.dot.W, Inf))")
end

# another test via ForwardDiff

using ForwardDiff: hessian

laplace_fwd(gfun, xx) = 
         [ tr( hessian(xx -> gfun(xx).dot.W[i], xx) ) 
            for i = 1:length(Δ1.dot.W) ]

Δ3 = laplace_fwd(g_ch3, xx)

Δ1.dot.W ≈ Δ3

