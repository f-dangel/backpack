"""Custom module example
=========================================

This tutorial shows how to support a custom module in a simple fashion. 
We will extend a custom module with the :py:class:`BatchGrad <backpack.extensions.BatchGrad>` extension as an example
for first-order extensions and the :py:class:`DiagGGNExact <backpack.extensions.DiagGGNExact>` extension as an example 
for second-order extensions.

Let's get the imports out of our way.
"""  # noqa: B950

import torch

from backpack import backpack, extend
from backpack.extensions import BatchGrad
from backpack.extensions.firstorder.base import FirstOrderModuleExtension

# make deterministic
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# Custom PyTorch module
# ---------------------
# In this example, we consider extending our own, very simplistic, layer.
# It scales the input by a scalar ``weight`` in a forward pass. Here is the
# layer class (see https://pytorch.org/docs/stable/notes/extending.html).


class ScaleModule(torch.nn.Module):
    """Defines the module."""

    def __init__(self, weight=2.0):
        """Store scalar weight.

        Args:
            weight(float, optional): Initial value for weight. Defaults to 2.0.
        """
        super(ScaleModule, self).__init__()

        self.weight = torch.nn.Parameter(torch.tensor([weight]))

    def forward(self, input):
        """Defines forward pass.

        Args:
            input(torch.Tensor): input

        Returns:
            torch.Tensor: product of input and weight
        """
        return input * self.weight


# %%
# You don't necessarily need to write a custom layer. Any PyTorch layer can be extended
# as described (it should be a :py:class:`torch.nn.Module <torch.nn.Module>`'s because
# BackPACK uses module hooks). If your functionality is not in a :py:class:`torch.nn.Module <torch.nn.Module>` yet,
# you can wrap it in a :py:class:`torch.nn.Module <torch.nn.Module>`.
#
# First-order extensions
# ----------------------
# First we focuses on `BackPACK's first-order extensions <https://docs.backpack.pt/en/master/extensions.html#first-order-extensions>`_.
# They don't backpropagate additional information and thus require less functionality to be
# implemented.

# Let's make BackPACK support computing individual gradients for ``ScaleModule``.
# This is done by the :py:class:`BatchGrad <backpack.extensions.BatchGrad>` extension.
# To support the new module, we need to create a module extension that implements
# how individual gradients are extracted with respect to ``ScaleModule``'s parameter.
#
# The module extension must implement methods named after the parameters passed to the
# constructor. In this case `weights`. For a module with additional parametes e.g. a `bias` additional methods named
# after these parameters have to be added. For parameter `bias` method `bias` is implemented.
#
# Here it goes.


class ScaleModuleBatchGrad(FirstOrderModuleExtension):
    """Extract individual gradients for ``ScaleModule``."""

    def __init__(self):
        """Store parameters for which individual gradients should be computed."""
        # specify parameter names
        super().__init__(params=["weight"])

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        """Extract individual gradients for ScaleModule's ``weight`` parameter.

        Args:
            ext(BatchGrad): extension that is used
            module(ScaleModule): module that performed forward pass
            g_inp(tuple[torch.Tensor]): input gradient tensors
            g_out(tuple[torch.Tensor]): output gradient tensors
            bpQuantities(None): additional quantities for second-order

        Returns:
            torch.Tensor: individual gradients
        """
        show_useful = True

        if show_useful:
            print("Useful quantities:")
            # output is saved under field output
            print("\tmodule.output.shape:", module.output.shape)
            # input i is saved under field input[i]
            print("\tmodule.input0.shape:", module.input0.shape)
            # gradient w.r.t output
            print("\tg_out[0].shape:     ", g_out[0].shape)

        # actual computation
        return (g_out[0] * module.input0).flatten(start_dim=1).sum(axis=1).unsqueeze(-1)


# %%
# Lastly, we need to register the mapping between layer (``ScaleModule``) and layer
# extension (``ScaleModuleBatchGrad``) in an instance of
# :py:class:`BatchGrad <backpack.extensions.BatchGrad>`.

# register module-computation mapping
extension = BatchGrad()
extension.set_module_extension(ScaleModule, ScaleModuleBatchGrad())

# %%
# That's it. We can now pass ``extension`` to a
# :py:class:`with backpack(...) <backpack.backpack>` context and compute individual
# gradients with respect to ``ScaleModule``'s ``weight`` parameter.

# %%
# Test custom module
# ------------------
# Here, we verify the custom module extension on a small net with random inputs.
# Let's create these.

batch_size = 10
batch_axis = 0
input_size = 4

inputs = torch.randn(batch_size, input_size, device=device)
targets = torch.randint(0, 2, (batch_size,), device=device)

reduction = ["mean", "sum"][1]
my_module = ScaleModule().to(device)
lossfunc = torch.nn.CrossEntropyLoss(reduction=reduction).to(device)

# %%
# .. note::
#     Results of ``"mean"`` and ``"sum"`` reduction differ by a scaling factor,
#     because the information backpropagated by PyTorch is scaled. This is documented at
#     https://docs.backpack.pt/en/master/extensions.html#backpack.extensions.BatchGrad.

# %%
# Individual gradients with PyTorch
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The following computes individual gradients by looping over individual samples and
# stacking their gradients.

grad_batch_autograd = []

for input_n, target_n in zip(
    inputs.split(1, dim=batch_axis), targets.split(1, dim=batch_axis)
):
    loss_n = lossfunc(my_module(input_n), target_n)
    grad_n = torch.autograd.grad(loss_n, [my_module.weight])[0]
    grad_batch_autograd.append(grad_n)

grad_batch_autograd = torch.stack(grad_batch_autograd)

print("weight.shape:             ", my_module.weight.shape)
print("grad_batch_autograd.shape:", grad_batch_autograd.shape)

# %%
# Individual gradients with BackPACK
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# BackPACK can compute individual gradients in a single backward pass.
# First, :py:func:`extend <backpack.extend>` model and loss function, then
# perform the backpropagation inside a
# :py:class:`with backpack(...) <backpack.backpack>` context.

my_module = extend(my_module)
lossfunc = extend(lossfunc)

loss = lossfunc(my_module(inputs), targets)

with backpack(extension):
    loss.backward()

grad_batch_backpack = my_module.weight.grad_batch

print("weight.shape:             ", my_module.weight.shape)
print("grad_batch_backpack.shape:", grad_batch_backpack.shape)

# %%
# Do the computation results match?

match = torch.allclose(grad_batch_autograd, grad_batch_backpack)

print(f"autograd and BackPACK individual gradients match? {match}")

if not match:
    raise AssertionError(
        "Individual gradients don't match:"
        + f"\n{grad_batch_autograd}\nvs.\n{grad_batch_backpack}"
    )

# %%
# Second-order Extension
# ----------------------
# Next, we focus on `BackPACK's second-order extensions <https://docs.backpack.pt/en/master/extensions.html#second-order-extensions>`_.
# They backpropagate additional information and thus require more functionality to be implemented and a more in depth
# understanding of BackPACK's internals and expert understanding of the metric
#
# Let's make BackPACK support computing the exact diagonal of the Gauss-Newton matrix for ``ScaleModule``.
#
# The first step is to have every part of the computation graph that is relevant for the Gauss-Newton matrix in
# :py:class:`torch.nn.Module <torch.nn.Module>` form. This is already the case for ``ScaleModule``. If this is not the
# case for your module you can wrap it in a :py:class:`torch.nn.Module <torch.nn.Module>`.
#
# The second step is to write a module extension that implements how the exact diagonal of the Gauss-Newton matrix is
# computed for ``ScaleModule``.
#
# To do this we need to understand the following about the extension:
#   1. The GGN is calculated by multiplying the Jacobian (w.r.t the parameters) with the Hessian of the loss function.
#      This is computed for every named parameter of the module.
#   2. The Hessian of the Loss function is generated by the :py:func:`extend <backpack.extend>` of the loss function
#      and backpropagated into the computation graph. The loss function has to be supported by BackPACK.
#   3. The value we want to backpropagate is the multiplication of the `input-output` Jacobian with the previous
#      backpropagated value.
#
# The to-be implemented second-order extension is for the GGN. Any other extension can be implemented in a similar
# fashion.
#
# GGN definitions
# ^^^^^^^^^^^^^^^
# Fist, the definition of the GGN:
# The GGN is calculated by multiplying the Jacobian (w.r.t the parameters) with the Hessian of the loss function.
#
# .. math::
#  \mathbf{G}(\theta) = (\mathbf{J}_\theta f_\theta(x))^T \; \nabla^2_{f_\theta(x^{(0)})} \ell (f_\theta(x^{(0)}), y) \; (\mathbf{J}_\theta f_\theta(x))
#
# The Jacobian (left & right of RHS) is the matrix of all first-order derivatives of the function (neural network) w.r.t. the parameters.
# The Hessian (center of RHS) is the matrix of all second-order derivatives of the loss function w.r.t. the neural network output.
# The GGN (LHS) will be a matrix with dim :math:`p \times p` where :math:`p` is the number of parameters. It is calculated
# w.r.t the parameters of the network. In the implementation we will have to split the computation for each named
# parameter, e.g. ``weight``, ``bias``, etc..
#
# If the loss function is convex, which is the case for many losses in ML, the following holds:
#
#
# .. math::
#  \exists S \in \mathbb{R}^{p \times p} \text{ s.t. } SS^T=\nabla^2_{f_\theta(x^{(0)})} \ell (f_\theta(x^{(0)}), y)
#
# There exists a decomposition of the Hessian into a multiplication of :math:`S` with its transpose.
# A corollary of this is that the GGN can be decomposed into a multiplication
# of :math:`V=(\mathbf{J}_\theta f_\theta(x))^T\;S` with its transpose:
#
# .. math::
#  \mathbf{G}(\theta) = V V^T = (\mathbf{J}_\theta f_\theta(x))^T\;S\;S^T\;(\mathbf{J}_\theta f_\theta(x))
#
# To compute the full GGN (or its diagonal) we can compute :math:`V` instead and multiply with :math:`V^T`.
#
# Calculations by Chain Rule
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# The Hessian and the required Jacobians are computed during the backward pass of the autograd engine using the chain rule.
# When using ANNs the autograd engine builds a representation of the ANN by using compositions of "atomic" operations.
# This is called computation graph. Consider the computation graph:
#
# .. image:: ../../images/comp_graph.jpg
#   :width: 75%
#   :align: center
#
# Each node in the graph represents a tensor. The arrows represent the flow of information and the computation associated
# with the incoming and outgoing tensors: :math:`f_{\theta^{(k)}}^{(k)}(x^{(k)}) = x^{(k+1)}`. The information is
# computed by the function---i.e. neural network layer---at the node.
#
# The parameter vector :math:`\theta` contains all parameters of the ANN and is composed of the stacked parameters of
# each layer of the neural network.
#
# .. math::
#  \theta = (\theta^{(1)}, \theta^{(2)}, \dots, \theta^{(l)})
#
# During the backward pass the autograd engine computes the derivatives of each function :math:`f_{\theta^(k)}^{(k)}`
# because the functions are multi-variate we call this the Jacobian :math:`\mathbf{J}_z y(z)` of the function :math:`y` w.r.t. :math:`z`.
# The full Jacobian of the neural network output w.r.t. the full parameter vector is the stacked Jacobian of each layer.
#
# .. math::
#  \mathbf{J}_\theta f_\theta(x) = (\mathbf{J}_{\theta^{(1)}} f_{\theta}(x^{(0)}), \mathbf{J}_{\theta^{(2)}} f_{\theta}(x^{(0)}), \dots, \mathbf{J}_{\theta^{(l)}} f_\theta(x^{(0)}))
#
# Due to the structure of the computation graph and the chain rule each Jacobian can be computed by multiplying the
# Jacobians against the information flow in the computation graph. For the path of interest:
#
# .. math::
#   p^{(k)} = ((\theta^{(k)} \rightarrow x^{(k)}), (x^{(k)} \rightarrow x^{(k+1)}), (x^{(k+1)} \rightarrow x^{(k+2)}),\dots, (x^{(l-1)} \rightarrow x^{(l)}))
#
# The Jacobian of this path is computed by chaining the local Jacobian of each computation:
#
# .. math::
#   \mathbf{J}_{\theta^{(k)}} f_{\theta}(x^{(0)}) = (\mathbf{J}_{x^{(l-1)}} f_\theta(x^{(0)}))\;\dots \; (\mathbf{J}_{x^{(k+2)}} x^{(k+1)})\;(\mathbf{J}_{x^{(k+1)}} x^{(k)})\;(\mathbf{J}_{\theta^{(k)}} x^{(k)})
#
# or equivalently:
#
# .. math::
#  \mathbf{J}_{\theta^{(k)}} f_{\theta}(x^{(0)}) = (\mathbf{J}_{\theta^{(k)}} x^{(k)})^T\;(\mathbf{J}_{x^{(k+1)}} x^{(k)})^T\;(\mathbf{J}_{x^{(k+2)}} x^{(k+1)})^T\;\dots \;(\mathbf{J}_{x^{(l-1)}} f_\theta(x^{(0)}))^T
#
# If we assume that we receive the Jacobian :math:`\mathbf{J}_{x^{(k)}} f_\theta (x^{(0)})` from the previous node in the graph we can focus the computation on the local Jacobian
# :math:`\mathbf{J}_{x^{(k-1)}} x^{(k)}` and :math:`\mathbf{J}_{\theta^{(k)}} x^{(k)}`. The current nodes backwarded Jacobian is then given by
#
# .. math::
#   (\mathbf{J}_{x^{(k-1)}} f_\theta (x^{(0)}))^T=(\mathbf{J}_{x^{(k-1)}} x^{(k)})^T\;(\mathbf{J}_{x^{(k)}} f_\theta (x^{(0)}))^T
#
# and the matrix :math:`V` is given by:
#
# .. math::
#   V=(\mathbf{J}_{\theta^{(k)}} x^{(k)})^T\;(\mathbf{J}_{x^{(k)}} f_\theta (x^{(0)}))^T
#
# In the implementation of the module extension we will implement the backwarded Jacobian in the ``backpropagate`` function and the matrix :math:`V` and the GGN in the ``weight`` function.
#
# Implementation
# ^^^^^^^^^^^^^^
#

from torch.nn.utils.convert_parameters import parameters_to_vector

# %%
# First some additional imports.
from backpack.extensions.module_extension import ModuleExtension
from backpack.extensions.secondorder.diag_ggn import DiagGGNExact
from backpack.hessianfree.ggnvp import ggn_vector_product
from backpack.utils.convert_parameters import vector_to_parameter_list

# %%
# The module extension must implement methods named after the parameters that are passed to the
# constructor. This is similar to the first-order extension. In addition it is necessary to implement the ``backpropagate`` function. This
# function is called by BackPACK during the backward pass and used to feed the Jacobians to later computations.


class ScaleModuleDiagGGNExact(ModuleExtension):
    """Extract diagonal of the Gauss-Newton matrix for ``ScaleModule``."""

    def __init__(self):
        """Store parameters for which individual gradients should be computed."""
        # specify parameter names
        super().__init__(params=["weight"])

    def backpropagate(self, ext, module, grad_inp, grad_out, bpQuantities):
        """Propagates second order information from the output to the input.

        Args:
            ext (DiagGGNExact): BackPACK extension.
            module (ScaleModule): module through which to perform backpropagation.
            grad_inp (Tuple[Tensor]): input gradients
            grad_out (Tuple[Tensor]): output gradients
            bpQuantities (Tuple[Tensor]): backpropagation information.
                Has shape [C, batch_size, D_out].

        Returns:
            The GGN diagonal's backpropagated quantity for the layer input.
            Has shape [C, batch_size, D_in].
        """
        assert ext.get_subsampling() is None

        # Layer:
        # - Input to the layer has shape [batch_size, D_in]
        # - Output of the layer has shape [batch_size, D_out]

        # Loss function:
        # - Neural networks prediction has shape [batch_size, C]

        # Quantity backpropagated by DiagGGNExact has shape [C, batch_size,
        # D_out] imagine this as a set of C vectors which all have the same
        # shape as the layer's output.

        # What we need to to now:
        # - Take each of the C vectors
        # - Multiply each of them with the layer's output-input Jacobian.
        #   The result of each VJP will have shape [batch_size, D_in]
        # - Stack them back together into a tensor of shape [C, batch_size, D_in]

        input0 = module.input0
        output = module.output
        weight = module.weight

        C = bpQuantities.shape[0]
        batch_size, D_in = input0.shape
        _, D_out = output.shape

        print("backpropagate: Useful quantities:")
        print(f"              module.output.shape: {output.shape}")
        print(f"              module.input.shape: {input0.shape}")
        print(f"              bpQuantities.shape: {bpQuantities.shape}")
        print(f"              returned.shape: {(C,) + input0.shape}")

        result = torch.zeros(
            (C, batch_size, D_in), device=input0.device, dtype=input0.dtype
        )

        # forward pass computation performs: X * weight ([batch_size, D_in] * [1] [batch_size, D_out=D_in])
        for c in range(C):
            result[c] = bpQuantities[c] * weight
        # or even simpler:
        # result = bpQuantities * weight
        assert result.shape == (C,) + input0.shape
        return result

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        """Calculate exact GGN diagonal for weight parameter.

        Args:
            ext (DiagGGNExact): BackPACK extension.
            module (ScaleModule): module through which to perform backpropagation.
            grad_inp (Tuple[Tensor]): input gradients
            grad_out (Tuple[Tensor]): output gradients
            bpQuantities (Tuple[Tensor]): backpropagation information. Has shape [C, batch_size, D_out].

        Returns:
            The GGN diagonal.
            Has shape [batch_size, H=1].
        """
        input0 = module.input0
        output = module.output
        weight = module.weight

        H = weight.shape[0]
        C = bpQuantities.shape[0]
        batch_size, D_in = input0.shape
        batch_size, D_out = output.shape
        print("weight: Useful quantities:")
        print(f"       module.output.shape {output.shape}")
        print(f"       module.input.shape {input0.shape}")
        print(f"       module.weight.shape {weight.shape}")
        print(f"       bpQuantities.shape {bpQuantities.shape}")
        print(f"       returned.shape {(batch_size,) + weight.shape}")

        # forward pass computation performs: X * weight ([batch_size, D_in] * [1] = [batch_size, D_out])
        # dimensions of J_theta_out is [batch_size, D_in, H]
        J_theta_out = input0.unsqueeze(-1)
        # change dimensions to [batch_size, C, D_in]
        # dimension of V is [batch_size, C, D_out] * [batch_size, D_out, H] = [batch_size, C, H]
        V = torch.matmul(bpQuantities.transpose(0, 1), J_theta_out)
        # compute diag(V^T * V)
        result = torch.matmul(V.transpose(1, 2), V).diagonal(dim1=1, dim2=2)
        assert result.shape == (batch_size, H)
        return result


# %%
# After we have implemented the module extension we need to register the mapping between layer (``ScaleModule``) and the
# layer extension (``ScaleModuleDiagGGNExact``) in an instance of :py:class:`DiagGGNExact <backpack.extensions.DiagGGNExact>`.

extension = DiagGGNExact()
extension.set_module_extension(ScaleModule, ScaleModuleDiagGGNExact())


# %%
# Testing the extension
# ^^^^^^^^^^^^^^^^^^^^^
# Here, we verify the custom module extension on a small net with random inputs as we have before.

# Calculate the extended ANN output and loss
# Calculate the GGN manually with internals of BackPACK
model = extend(my_module)
lossfunc = extend(lossfunc)

params = list(my_module.parameters())

ggn_dim = sum(p.numel() for p in params)
diag_ggn_flat = torch.zeros(
    batch_size * ggn_dim, device=inputs.device, dtype=inputs.dtype
)
# looping explicitly over the batch dimension
for b in range(batch_size):
    outputs = my_module(inputs[b])
    loss = lossfunc(outputs, targets[b])

    for d in range(ggn_dim):
        # create unit vector d
        e_d = torch.zeros(ggn_dim, device=inputs.device, dtype=inputs.dtype)
        e_d[d] = 1.0
        e_d = vector_to_parameter_list(e_d, params)

        # multiply GGN onto the unit vector -> get back column d of the GGN
        ggn_e_d = ggn_vector_product(loss, outputs, model, e_d)
        # flatten
        ggn_e_d = parameters_to_vector(ggn_e_d)

        # extract the d-th entry (which is on the GGN's diagonal)
        diag_ggn_flat[b * ggn_dim + d] = ggn_e_d[d]

print(f"Tr(GGN): {diag_ggn_flat.sum():.3f}")

# Calculate the extended ANN output and loss
model = extend(my_module)
lossfunc = extend(lossfunc)

outputs = my_module(inputs)
loss = lossfunc(outputs, targets)

with backpack(extension):
    loss.backward()

# compare with the ground truth
diag_ggn_flat_backpack = parameters_to_vector(
    [p.diag_ggn_exact for p in model.parameters()]
)
print(f"Tr(GGN, BackPACK): {diag_ggn_flat_backpack.sum():.3f}")
match = torch.allclose(diag_ggn_flat, diag_ggn_flat_backpack)

print(f"Do manual and BackPACK GGN match? {match}")

if not match:
    raise AssertionError(
        "exact GGN diagonal does not match:"
        + f"\n{grad_batch_autograd}\nvs.\n{grad_batch_backpack}"
    )
