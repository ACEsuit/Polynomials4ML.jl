module ObjectPools

export acquire!, 
       release!, 
       ArrayCache

using Base.Threads: threadid, nthreads
using DataStructures: Stack 

struct ArrayCache{T, N}
   cache::Vector{Stack{Array{T, N}}}
end

struct CachedArray{N, T} <: AbstractArray{T, N} 
   A::Array{T, N}
   pool::ArrayCache{T}
end

release!(A::Any) = nothing 

# here we are assuming that that array doesn't get passed around different 
# threads but will be released in the same thread it was acquired in.
# if this fails, the code remains *correct* but might become inefficient.
release!(pA::CachedArray) = release!(pA.pool, pA)

using Base: @propagate_inbounds

@propagate_inbounds function Base.getindex(pA::CachedArray, I...) 
   @boundscheck checkbounds(pA.A, I...)
   @inbounds pA.A[I...]
end

@propagate_inbounds function Base.setindex!(pA::CachedArray, val, I...)
   @boundscheck checkbounds(pA.A, I...)
   @inbounds pA.A[I...] = val
end

Base.length(pA::CachedArray) = length(pA.A)

Base.eltype(pA::CachedArray) = eltype(pA.A)

Base.size(pA::CachedArray, args...) = size(pA.A, args...)

Base.parent(pA::CachedArray) = pA.A


function ArrayCache(T, N)
   nt = nthreads() 
   cache = [ Stack{Array{T, N}}() for _1=:nt ] 
   return ArrayCache{T, N}(cache)
end


acquire!(c::ArrayCache{T, 1}, len::Integer, ::Type{T}, ) where {T} = 
         acquire!(c, len)

acquire!(c::ArrayCache{T}, len::Integer, ::Type{S}) where {T, S} =
         Vector{S}(undef, len)

function acquire!(c::ArrayCache{T}, len::Integer) where {T}
   stack = c.cache[threadid()]
   if isempty(stack)
      A = Vector{T}(undef, len)
   else 
      A = pop!(stack)
      resize!(A, len)
   end
   return CachedArray(A, c)
end

release!(c::ArrayCache, cA::CachedArray{1}) = 
      push!(c.vecs[threadid()], cA.A)


# ## Alternative Array Cache Implementation 


# struct GenArrayCache
#    vecs::Vector{Stack{Vector{UInt8}}}
# end

# # struct GenCachedArray{T, N, AT <: AbstractArray{T, N}} <: AbstractArray{T, N}
# #    A::AT
# #    pool::GenArrayCache
# # end

# using UnsafeArrays

# struct GenCachedArray{T, N} <: AbstractArray{T, N}
#    A::UnsafeArray{T, N}
#    _A::Vector{UInt8}
#    pool::GenArrayCache
# end


# # release!(A::Any) = nothing 
# release!(pA::GenCachedArray) = release!(pA.pool, pA)

# using Base: @propagate_inbounds

# @propagate_inbounds function Base.getindex(pA::GenCachedArray, I...) 
#    @boundscheck checkbounds(pA.A, I...)
#    @inbounds pA.A[I...]
# end

# @propagate_inbounds function Base.setindex!(pA::GenCachedArray, val, I...)
#    @boundscheck checkbounds(pA.A, I...)
#    @inbounds pA.A[I...] = val
# end

# # Base.getindex(pA::CachedArray, args...) = getindex(pA.A, args...)

# # Base.setindex!(pA::CachedArray, args...) = setindex!(pA.A, args...)

# Base.length(pA::GenCachedArray) = length(pA.A)

# Base.eltype(pA::GenCachedArray) = eltype(pA.A)

# Base.size(pA::GenCachedArray, args...) = size(pA.A, args...)

# Base.parent(pA::GenCachedArray) = pA.A


# function GenArrayCache() where {T} 
#    nt = nthreads()
#    vecs = [ Stack{Vector{UInt8}}() for _=1:nt ]
#    return GenArrayCache(vecs)
# end


# function acquire!(c::GenArrayCache, ::Type{T}, len::Integer) where {T} 
#    szofT = sizeof(T)
#    szofA = len * szofT
#    stack = c.vecs[threadid()]
#    if isempty(stack)
#       _A = Vector{UInt8}(undef, szofA)
#    else 
#       _A = pop!(stack)
#       resize!(_A, szofA)
#    end
#    # A = reinterpret(T, _A)
#    # return GenCachedArray(A, c)

#    ptr = Base.unsafe_convert(Ptr{T}, _A)
#    # A = Base.unsafe_wrap(Array, ptr, len)
#    A = UnsafeArray(ptr, (len,))
#    return GenCachedArray(A, _A, c)
# end

# # release!(c::GenArrayCache, cA::GenCachedArray) = 
# #       push!(c.vecs[threadid()], parent(cA.A))

# release!(c::GenArrayCache, cA::GenCachedArray) = 
#       push!(c.vecs[threadid()], cA._A)

end