"""Hessian backpropagation for Hessian block-diagonal approximations.

Implementation of the Hessian backpropagation procedure (HBP) proposed in

* F. Dangel, P. Hennig: "A Modular Approach to Block-diagonal Hessian
  Approximations for Second-order Optimization Methods" (2019),
  URL: https://arxiv.org/pdf/1902.01813.pdf

HBP allows to compute diagonal blocks of the Hessian by an additional
backward pass. For more information and notation, please refer to the
links provided above.

# Naming convention:
If you want to use the HBP extension of layer torch.nn.LayerX, the
HBP implementation of that layer is given by the class HBPLayerX.
For instance, the linear layer provided by torch.nn.Linear is
implemented by hbp.linear.HBPLinear.

# Note
HBP works for feedforward neural networks.

# HBP
A module $z = f(x, \theta)$ mapping an input $x$ to an output $z$
by means of parameters $\theta$ has to implement two basic operations
for being integrated in the framework of HBP:

Given the Hessian $\mathcal{H}z$ of the loss function with respect to
the module's output $z$,

1.) Compute the Hessian $\mathcal{H} \theta$ of the loss w.r.t. the
 module parameters $\theta$.
2.) Compute the batch average of the Hessian $\mathcal{H} x$ of the
 loss w.r.t. the module's input

After propagating back a Hessian through the graph, each parameter of
the net holds a function implementing matrix-vector products with in
the hvp attribute.

# Implementation details
HBP requires the storage of additional quantities during the forward
and backward pass of gradients. In PyTorch, this is implemented by
decorating a module with hooks. The decorator hbp_decorate implemented
in hbp.module takes a subclass of torch.nn.Module and adds the
functionality to install hooks that store the required quantities.

"""
