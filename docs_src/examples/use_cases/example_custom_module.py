"""Custom module example
=========================================

This tutorial shows how to support a custom module in a simple fashion.
"""
import copy

import torch
from torch.nn import CrossEntropyLoss

from backpack import backpack, extend
from backpack.extensions import BatchGrad
from backpack.extensions.firstorder.base import FirstOrderModuleExtension

# make deterministic
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# Adding custom module to BackPACK
# -------------------------
# Define a custom module (https://pytorch.org/docs/stable/notes/extending.html).
#
# This example scales the input by a factor named "weight".
class ScaleModule(torch.nn.Module):
    """Defines the module."""

    def __init__(self, weight=2.0):
        """Initializes scale module.

        Args:
            weight(float, optional): Initial value for weight. Defaults to 2.0.
        """
        super(ScaleModule, self).__init__()

        self.weight = torch.nn.Parameter(torch.tensor(weight))

    def forward(self, input):
        """Defines forward pass.

        Args:
            input(torch.Tensor): input

        Returns:
            torch.Tensor: product of input and weight
        """
        return input * self.weight


# %%
# To support batch gradients of this module in BackPACK, we write an extension.
# This extension should implement methods named after the parameters
# that calculate the batch gradients.
#
# Finally, we add the class to BackPACK.
class ScaleModuleBatchGrad(FirstOrderModuleExtension):
    """Extract indiviual gradients for ``ScaleModule``."""

    def __init__(self):
        """Initializes scale module batch extension."""
        # specify the names of the parameters
        super().__init__(params=["weight"])

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        """Extract individual gradients for ScaleModule's weight parameter.

        Args:
            ext(BatchGrad): extension that is used
            module(ScaleModule): module that performed forward pass
            g_inp(tuple[torch.Tensor]): input gradient tensors
            g_out(tuple[torch.Tensor]): output gradient tensors
            bpQuantities(None): additional quantities for second order

        Returns:
            torch.Tensor: individual gradients
        """
        # useful available quantities are
        # output is saved under field output
        # print("module.output", module.output)
        # input i is saved under field input[i]
        # print("module.input0", module.input0)
        # gradient of output
        # print("g_out[0]", g_out[0])
        return (g_out[0] * module.input0).reshape((g_out[0].shape[0], -1)).sum(axis=1)


# add the class to backpack
BatchGrad.add_module_extension(ScaleModule, ScaleModuleBatchGrad())

# %%
# Testing custom module
# -------------------------
# Create some random data and define a function.
#
# Note, that using "mean" instead of "sum", leads to a within-batch dependency.
# This scales the vectors backpropagated by PyTorch by ``1 / batch_size``.
# This scaling peculiarity is also documented here
# (https://docs.backpack.pt/en/master/extensions.html#backpack.extensions.BatchGrad).
batch_size = 10
input_size = 4
input = torch.randn(batch_size, input_size)
target = torch.randint(0, 2, (batch_size,))

reduction = ["mean", "sum"][1]
my_module = ScaleModule()
lossfunc = CrossEntropyLoss()
lossfunc.reduction = reduction

# %%
# The normal backward pass.
loss = lossfunc(my_module(input), target)

loss.backward()

for name, param in my_module.named_parameters():
    print(name)
    print(".grad.shape:     ", param.grad.shape)

# %%
# Backward with backpack.
my_module_ext = extend(copy.deepcopy(my_module))
lossfunc_ext = extend(copy.deepcopy(lossfunc))
my_module_ext.zero_grad()
loss = lossfunc_ext(my_module_ext(input), target)
print("loss", loss)

with backpack(BatchGrad()):
    loss.backward()

for name, param in my_module_ext.named_parameters():
    print(name)
    print(".grad.shape:         ", param.grad.shape)
    print(".grad_batch.shape:   ", param.grad_batch.shape)

print(
    "Does batch gradient match with individual gradients?",
    torch.allclose(
        my_module_ext.weight.grad, my_module_ext.weight.grad_batch.sum(axis=0)
    ),
)

# ensuring this test runs
assert torch.allclose(
    my_module_ext.weight.grad, my_module_ext.weight.grad_batch.sum(axis=0)
)

# %%
# Calculate the individual gradients with autograd and compare to BackPACK.
grad_batch_backpack = my_module_ext.weight.grad_batch
grad_batch_autograd = torch.zeros(grad_batch_backpack.shape)
for n in range(batch_size):
    my_module.zero_grad()
    loss = lossfunc(my_module(input[n].unsqueeze(0)), target[n].unsqueeze(0))
    loss.backward()
    grad_batch_autograd[n] = my_module.weight.grad

print("grad_batch_autograd.shape", grad_batch_autograd.shape)

print(
    "Do autograd and backpack individual gradients match?",
    torch.allclose(grad_batch_autograd, grad_batch_backpack),
)

# ensuring this test runs
assert torch.allclose(grad_batch_autograd, grad_batch_backpack)
