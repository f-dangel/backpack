"""Backpropagation extensions in PyTorch.

Problem: Implementing optimization algorithms that require access to
additional information apart from the batch sum of parameter gradients
is hard.

This package provides a way to modify the forward and backward pass of
a module in PyTorch (a subclass of torch.nn.Module).

We do so by providing a decorator (bpexts.decorator.decorate) which extends
 a given PyTorch layer by adding functions to install and remove hooks and
 buffers which can be used to extend the forward/backward behavior.

Note: PyTorch allows the installation of module hooks/buffers out of the
 box, but we chose to separate the hooks/buffers. This leads to a cleaner
implementation whenever the user wants to install additional hooks/buffers
on top using the default functionality of PyTorch.

# Subpackages

1) bpexts.hbp

Implements the Hessian backpropagation (HBP) procedure. This is a technique
to obtain approximations to the block diagonal of the Hessian. It is
outlined in

* F. Dangel, P. Hennig: "A Modular Approach to Block-diagonal Hessian
  Approximations for Second-order Optimization Methods" (2019),
  URL: https://arxiv.org/pdf/1902.01813.pdf

2) bpexts.optim

Implements optimizers that make use of the additional quantities from the
backward extension.


3) bpexts.gradient

Compute gradients for each sample in a batch. Usually, the gradients for
 each data point in the batch are summed up. For instance, this facilitates
the implementation of variance-adapted optimization methods

4) bpexts.hessian

Brute-force computation of Hessian matrices by twice application of automatic
differentiation.


5) bpexts.sumgradsquared

(Coming soon)
"""
