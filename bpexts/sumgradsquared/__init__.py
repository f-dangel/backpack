"""Computation of Sum-Grad-Squared (SGS)

For each parameter of the model, computes the sum (over the batch) of the gradients squared.

Useful for computing
* the diagonal of the Empirical Fisher
* Monte Carlo estimates of the diagonal of the Fisher
* the variance of the stochastic gradients.

"""
