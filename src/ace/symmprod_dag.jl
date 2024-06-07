
import Base: ==, length

using Combinatorics: combinations, partitions

const BinDagNode = Tuple{Int, Int}

export SparseSymmProdDAG

"""
`struct SparseSymmProdDAG` : alternative (recursive) implementation of 
`SparseSymmProd`. This has better theoretical performance for high correlation 
orders. 

The potential downside is that it inserts auxiliary basis functions into the 
basis. This means, that the specification of the output will be different 
from the specification that is used to construct the basis. To that end, the 
field `projection` can be used to reduce it back to the original spec. E.g., 
```julia
basis = SparseSymmProd(spec)
basis_dag = SparseSymmProdDAG(spec)
A = randn(nA)
basis(A) ≈ basis_dag(A)[basis_dag.projection]   # true
```

However, the field `projection` is used only for information, and not 
to actually reduce the output. One could of course use it to compose 
the output with a projection matrix.
"""
struct SparseSymmProdDAG <: AbstractP4MLTensor
   nodes::Vector{BinDagNode}
   has0::Bool
   num1::Int
   numstore::Int
   projection::Vector{Int}
   # ---- temps
   @reqfields
end

# warning: if SparseSymmProdDAG is an extended basis, then `length` will be 
# the extended length and not the length of the actual basis. 

length(dag::SparseSymmProdDAG) = length(dag.nodes)

# ==(dag1::SparseSymmProdDAG, dag2::SparseSymmProdDAG) = ACE1._allfieldsequal(dag1, dag2)

SparseSymmProdDAG() = SparseSymmProdDAG(Vector{BinDagNode}(undef, 0), false, 0, 0)

SparseSymmProdDAG(nodes, has0, num1, numstore, proj)   = 
         SparseSymmProdDAG(nodes, has0, num1, numstore, proj, 
                              _make_reqfields()...)


# # -------------- FIO

# # TODO: maybe there is a cleverer way to do this, for now this is just
# #       a quick hack to make sure it can be read without any ambiguity

# write_dict(gr::SparseSymmProdDAG{T, TI}) where {T <: Real, TI <: Integer} =
#    Dict( "__id__" => "ACE1_SparseSymmProdDAG",
#          "T" => write_dict(T),
#          "TI" => write_dict(TI),
#          "nodes1" => [ n[1] for n in gr.nodes ],
#          "nodes2" => [ n[2] for n in gr.nodes ],
#          "num1" => gr.num1,
#          "numstore" => gr.numstore
#       )


# function read_dict(::Val{:ACE1_SparseSymmProdDAG}, D::Dict)
#    T = read_dict(D["T"])
#    TI = read_dict(D["TI"])
#    @assert T <: Real
#    @assert TI <: Integer
#    return SparseSymmProdDAG{T, TI}(
#       collect(zip(D["nodes1"], D["nodes2"])),
#       D["num1"],
#       D["numstore"]
#    )
# end


# ---------------------------------------------------------------------
#   partition generator

_score_partition(p) = isempty(p) ? Inf : (1e9 * length(p) + maximum(p))

function _get_ns(p, specnew, specnew_dict)
   out = Vector{Int}(undef, length(p))
   for (i, kk_) in enumerate(p)
      if haskey(specnew_dict, kk_)
         out[i] = specnew_dict[kk_]
      else
         return Int[]
      end
   end
   return out
end


function _find_partition(kk, specnew, specnew_dict)
   has0 = (length(specnew[1]) == 0)
   worstp = _get_ns([ [k] for k in kk ], specnew, specnew_dict)
   @assert worstp == has0 .+ kk
   bestp = worstp
   bestscore = _score_partition(bestp)

   for ip in partitions(1:length(kk))
      p = _get_ns([ kk[i] for i in ip ], specnew, specnew_dict)
      score = _score_partition(p)
      if !isempty(p) && score < bestscore
         bestp = p
         bestscore = score
      end
   end

   return bestp
end


# return value is the number of fake nodes added to the dag
function _insert_partition!(nodes, specnew, specnew_dict,
                            kk, p,
                            ikk, specN)
   if length(p) == 2
      newnode = BinDagNode((p[1], p[2]))
      push!(nodes, newnode)
      push!(specnew, kk)
      specnew_dict[kk] = length(specnew)
      return 0
   else
      # reduce the partition by pushing a new node
      push!(nodes, BinDagNode((p[1], p[2])))
      kk1 = sort(vcat(specnew[p[1]], specnew[p[2]]))
      push!(specnew, kk1)
      specnew_dict[kk1] = length(specnew)
      # and now recurse with the reduced partition
      return 1 + _insert_partition!(nodes, specnew, specnew_dict,
                         kk, vcat( [length(nodes)], p[3:end] ),
                         ikk, specN)
   end
end

"""
Construct the DAG used to evaluate an AA basis and returns it as a `SparseSymmProdDAG`

Arguments
* `spec` : AA basis specification, list of vectors of integers / indices pointing into A 

Kwargs: 
* `filter = _-> true` : 
* `verbose = false` : print some information about the 
"""
function SparseSymmProdDAG(spec::AbstractVector; 
                           filter = _->true, 
                           verbose = false)
   @assert issorted(length.(spec))
   @assert all(issorted, spec)
   # we need to separate them into 0-corr, 1-corr and N-corr
   has0 = (length(spec[1]) == 0)
   spec1 = spec[ length.(spec) .== 1 ]
   IN = (length(spec1)+1+has0):length(spec)
   specN = spec[IN]

   # start assembling the dag
   nodes = BinDagNode[]
   sizehint!(nodes, length(spec))
   specnew = Vector{Int}[]
   specnew_dict = Dict{Vector{Int}, Int}()
   sizehint!(specnew, length(spec))

   # add the zero-correlation into the dag 
   if has0
      push!(nodes, BinDagNode((0, 0)))
      push!(specnew, Int[])
      specnew_dict[ Int[] ] = length(specnew)
   end

   # add the full 1-particle basis (N=1) into the dag
   _mymax(vv) = length(vv) == 0 ? 0 : maximum(vv) # hack due to init missing for SVectors
   num1 = maximum( _mymax(vv) for vv in spec )
   for i = 1:num1
      push!(nodes, BinDagNode((i+has0, 0)))
      push!(specnew, [i])
      specnew_dict[ [i] ] = length(specnew)
   end

   # now we can construct the rest
   extranodes = 0
   for (ikk, kk) in enumerate(specN)
      # find a good partition of kk
      p = _find_partition(kk, specnew, specnew_dict)
      extranodes += _insert_partition!(nodes, specnew, specnew_dict,
                                       kk, p, ikk, specN)
   end

   verbose && @info("Extra nodes inserted into the dag: $extranodes")
   numstore = length(nodes)
   num1old = num1

   projection = [ specnew_dict[vv] for vv in spec ]

   # re-organise the dag layout to minimise numstore
   # nodesfinal, num1, numstore = _reorder_dag!(nodes)

   return SparseSymmProdDAG(nodes, has0, num1, numstore, projection)
end


# TODO: this is currently not used; first need to add the functionality to 
#       to update specnew_dict i.e. the inverse mapping of specnew
function _reorder_dag!(nodes::Vector{BinDagNode})
   # collect all AA indices that are used anywhere in the dag
   newinds = zeros(Int, length(nodes))
   newnodes = BinDagNode[]

   # inds2 = stage-2 indices, i.e. temporary storage
   # inds3 = stage-3 indices, i.e. no intermediate storage
   inds2 = sort(unique([[ n[1] for n in nodes ]; [n[2] for n in nodes]]))

   # first add all 1p nodes
   for i = 1:length(nodes)
      n = nodes[i]
      if (n[2] == 0) ## && ((c != 0) || (i in inds2))
         @assert n[1] == i
         newinds[i] = i
         push!(newnodes, n)
      end
   end
   num1 = length(newnodes)

   # next add the remaining dependent nodes
   for i = 1:length(nodes)
      n = nodes[i]
      # not 1p basis && dependent node
      if (n[2] != 0) && (i in inds2)
         push!(newnodes, BinDagNode((newinds[n[1]], newinds[n[2]])))
         newinds[i] = length(newnodes)
      end
   end
   numstore = length(newnodes)

   # now go through one more time and add the independent nodes
   for i = 1:length(nodes)
      n = nodes[i]
      if (n[2] != 0) && (newinds[i] == 0)
         push!(newnodes, BinDagNode((newinds[n[1]], newinds[n[2]])))
         newinds[i] = length(newnodes)
      end
   end

   return newnodes, num1, numstore
end



# ------------------------------------------------------------------
# reconstruct the specification from the DAG ... 


function reconstruct_spec(dag::SparseSymmProdDAG)
   spec = Vector{Int}[]
   has0 = dag.has0
   for i = 1:length(dag.nodes)
      n = dag.nodes[i]
      if n[1] == n[2] == 0 
         bb = Int[]
         push!(spec, bb)
      elseif n[2] == 0
         bb = Int[ n[1] - has0, ]
         push!(spec, bb)
      else
         bb = sort(vcat(spec[n[1]], spec[n[2]]))
         push!(spec, bb)
      end
   end
   return spec
end


