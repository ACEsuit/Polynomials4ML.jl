# ## Polynomial Regression with P4ML 
#
# This tutorial show how to use the P4ML package to perform a naive polynomial 
# regression. This is not really the intended use-case of the package, but it 
# serves to illustrate some basic usage and functionality.
#

using Polynomials4ML 

### Example 1: Univeriate Polynomial Regression 
#
# First, we specify a target function, which we will try to approximate.

f1(x) = 1 / (1 + 10 * x^2);

# Suppose we are given data in the form of argument, function value pairs, 
# with arguments sampled uniformly from the interval [-1, 1].

N = 1_000 ;
Xtrain = 2 * rand(N) .- 1;
Ytrain = f1.(Xtrain);

# We want to solve the least squares problem 
# ```math 
#   \min_p \sum_{i = 1}^{N} | y_i - p(x_i) |^2
# ```
# for a polynomial $p$. Because ``f_1`` is analytic, we use a degree 
# approximately ``\sqrt{N}``. Since the distribution of the 
# samples is uniform, the Legrendre polynomials are likely a good choice. 

basis = legendre_basis(ceil(Int, sqrt(N)))
c  = basis(Xtrain) \ Ytrain 
p = x -> sum(c .* basis(x)) 

# A quick statistical test error 

Xtest = 2 * rand(N) .- 1
Ytest = f1.(Xtest)
println("Test RMSE: ", sqrt(sum(abs2, Ytest - p.(Xtest)) / N))

#
### Example 2: Polynomial Regression on the Sphere
#
#  [todo]



