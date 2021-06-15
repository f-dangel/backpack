"""Custom module example
=========================================

This tutorial shows how to support a custom module in a simple fashion.
We focus on `BackPACK's first-order extensions <https://docs.backpack.pt/en/master/extensions.html#first-order-extensions>`_.
They don't backpropagate additional information and thus require less functionality be
implemented.

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
# BackPACK uses module hooks).
#
# Custom module extension
# -----------------------
# Let's make BackPACK support computing individual gradients for ``ScaleModule``.
# This is done by the :py:class:`BatchGrad <backpack.extensions.BatchGrad>` extension.
# To support the new module, we need to create a module extension that implements
# how individual gradients are extracted with respect to ``ScaleModule``'s parameter.
#
# The module extension must implement methods named after the parameters passed to the
# constructor. Here it goes.


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
