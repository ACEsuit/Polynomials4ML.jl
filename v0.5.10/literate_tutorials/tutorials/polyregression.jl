# ### Linear Polynomial Regression 
#
# This tutorial show how to use the P4ML package to perform a naive polynomial 
# regression. This is not really the intended use-case of the package, but it 
# serves to illustrate some basic usage and functionality. 
#

using Polynomials4ML, LinearAlgebra, StaticArrays

# ### Example 1: Univariate Polynomial Regression 
#
# First, we specify a target function, which we will try to approximate.

f1(x) = 1 / (1 + 10 * x^2);

# Suppose we are given data in the form of argument, function value pairs, 
# with arguments sampled uniformly from the interval [-1, 1].

N = 1_000
Xtrain = 2 * rand(N) .- 1
Ytrain = f1.(Xtrain);

# We want to solve the least squares problem 
# ```math 
#   \min_p \sum_{i = 1}^{N} | y_i - p(x_i) |^2
# ```
# for a polynomial $p$. Because ``f_1`` is analytic, we use a degree 
# approximately ``\sqrt{N}``. Since the distribution of the 
# samples is uniform, the Legrendre polynomials are likely a good choice. 

basis = legendre_basis(ceil(Int, sqrt(N)))
c = basis(Xtrain) \ Ytrain 
p = x -> sum(c .* basis(x));

# A quick statistical test error 

Xtest = 2 * rand(N) .- 1
Ytest = f1.(Xtest)
println("Test RMSE: ", norm(Ytest - p.(Xtest)) / sqrt(N))

#
# ### Example 2: Polynomial Regression on the Sphere
#
# As a second example we consider a function defined on the unit sphere, 

f2(x) = 1 / (1 + 10 * norm(x - SA[1.0,0.0,0.0])^2);

# where `x` is now an `SVector{3, Float64}` with `norm(x) == 1`. 
# We generate again uniform samples on the sphere. 

N = 1_000 
X = [ (x = randn(SVector{3, Float64}); x/norm(x)) for i = 1:N ]
Y = f2.(X);

# In this case, spherical harmonics are natural basis functions. P4ML implements 
# both real and complex spherical harmonics. Since the target function is 
# real, we choose the real basis. 

basis = real_sphericalharmonics(9)
@show length(basis);

# We see that we now have far more basis functions per degree and therefore 
# cannot use as high a degree as before (unless we give ourselves more data). 
# Otherwise we can proceed exactly as above. 

c = basis(X) \ Y
p = x -> sum(c .* basis(x));

# We can test the RMSE again. 

Xtest = [ (x = randn(SVector{3, Float64}); x/norm(x)) for i = 1:N ]
Ytest = f2.(Xtest)
println("Test RMSE: ", norm(Ytest - p.(Xtest)) / sqrt(N));


