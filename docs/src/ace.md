# Cluster Expansion

The Atomic Cluster Expansion (ACE) and its relatives and derivatives are one of the main reasons we wrote P4ML. 
ACE can be thought of as a highly efficient but systematic scheme to construct polynomial approximations of permutation invariant functions or of multi-set functions, that may in addition be equivariant under some Lie group action. The same model components can also be used to construct euquivariant message passing networks such as E3NN. 

Here, we focus only on the permutation invariance. This brief introduction documents 
- [`PooledSparseProduct`](@ref)
- [`SparseSymmProd`](@ref)
- [`SparseSymmProdDAG`](@ref) 
This document only provides background. For usage, we refer to the inline documention. 



Suppose we have such a multi-set function
```math 
    f( [x_j]_j )
```
where ``[x_j]_j`` denotes a multi-set and each set element ``x_j \in \mathbb{R}^d`` is a vector (though it could also be a more general object). In the cluster expansion, one starts with an embedding of ``x_j`` into an abstract space ``V`` given by a one-particle basis ``\phi_k``, i.e., 
```math 
    x_j \mapsto (\phi_k(x_j))_k \in V
```
The polynomial bases implemented in P4ML (orthogonal polynomials, trigonometric polynomials, spherical harmonics, etc) can be used to construct those one-particle embeddings. Here, we keep ``\phi_k, V`` abstract so that we can focus on the two key model components that are unique to ACE. 

## Fused Product and Pooling

Given a one-particle embedding, we obtain an embedding of the multiset ``[x_j]_j`` into the same space ``V``, 
```math
   A = (A_k)_k \quad \text{where} \quad 
   A_k( [x_j]_j ) = \sum_j \phi_k(x_j).
```
This is implemented using a fused sparse product and pooling operation:
- [`PooledSparseProduct`](@ref)

This is a model layer that merged the pooling operation ``\sum_j`` with taking the product that is normally involved when evaluating ``\phi_k``. When ``x_j`` are a vector, then ``\phi_k(x_j)`` is normally a tensor product, e.g., ``\phi_{nlm}({\bm r}) = R_{nl}(r) Y_l^m(\hat{{\bm r}})``. 

## N-Correlations (Symmetric Product)

After forming the embedding ``A`` the cluster expansion models then for symmetric tensor products, or, ``N``-correlations, 
```math
   {\bm A}_{k_1 \dots k_N} = \prod_{t = 1}^N A_{k_t}.
```
This can also be seen as a symmetric rank-1 product ``A \otimes \cdots \otimes A``. 

If the ``\phi_k`` form a complete basis of one-particle functions, then the ``{\bm A}_{\bm k}`` form a complete basis of multi-set functions. E.g., one can then expand a permutation-invariant function as 
```math
   f([x_j]_j) = \sum_{\bm k \in \mathcal{K}} c_{\bm k} {\bm A}_{\bm k}([x_j]_j),
```
where ``\mathcal{K}`` is a list of ordered ``(k_t)`` tuples specifying the basis. 

There are two different implementations of this model component in P4ML: 
- [`SparseSymmProd`](@ref) : a naive implementation that is performant for relatively small ``N``
- [`SparseSymmProdDAG`](@ref) : a more efficient recursive implementation, behaves slightly differently, to be used with care




## References

1. Ralf Drautz. Atomic cluster expansion for accurate and transferable interatomic potentials. Phys. Rev. B Condens. Matter, 99(1):014104, January 2019
2. Dusson, G., Bachmayr, M., Csányi, G., Drautz, R., Etter, S., van der Oord, C., & Ortner, C. (2022). [Atomic cluster expansion: Completeness, efficiency and stability](https://arxiv.org/pdf/1911.03550.pdf). Journal of Computational Physics, 454, 110946.
3. Yury Lysogorskiy, Cas van der Oord, Anton Bochkarev, Sarath Menon, Matteo Rinaldi, Thomas Hammerschmidt, Matous Mrovec, Aidan Thompson, G ́abor Cs ́anyi, Christoph Ortner, and Ralf Drautz. Performant implementation of the atomic cluster expansion (pace) and application to copper and silicon. npj Computational Materials, 7(1):97, 2021.
