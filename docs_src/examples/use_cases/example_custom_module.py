"""Custom module example
=========================================

This tutorial shows how to support a custom module in a simple fashion.
"""

# %%
#
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
    """Defines the module based on ScaleFunction."""

    def __init__(self, input_size=(1,), weight=2.0):
        """Initializes scale module.

        Args:
            input_size: size of input is equivalent to output size
            weight(float): Initial value for weight. Defaults to 2.0.
        """
        super(ScaleModule, self).__init__()
        self.input_size = input_size

        self.weight = torch.nn.Parameter(torch.tensor(weight))

    def forward(self, input):
        """Defines forward pass based on ScaleFunction.

        Args:
            input(torch.Tensor): input

        Returns:
            torch.Tensor: Result from ScaleFunction
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
        return (g_out[0] * module.input0).sum(axis=1)


# add the class to backpack
BatchGrad.add_module_extension(ScaleModule, ScaleModuleBatchGrad())


# %%
# Testing custom module
# -------------------------
# Create some random data and define a function.
#
# Note that using "mean" instead of "sum", leads to a within-batch dependency.
# This alters the individual gradients in autograd (see later) by a factor batch_size.
batch_size = 10
input_size = 4
input = torch.randn(batch_size, input_size)
target = torch.randint(0, 2, (batch_size,))

reduction = ["mean", "sum"][1]
scaleModule = ScaleModule(input_size=(input_size,))
lossfunc = CrossEntropyLoss()
lossfunc.reduction = reduction

# %%
# The normal backward pass.
loss = lossfunc(scaleModule(input), target)

loss.backward()

for param in scaleModule.parameters():
    print("batch gradient", param.grad)


# %%
# Backward with backpack.
scaleModule_ext = extend(scaleModule)
lossfunc_ext = extend(lossfunc)
scaleModule_ext.zero_grad()
loss = lossfunc_ext(scaleModule_ext(input), target)
print("loss", loss)

with backpack(BatchGrad()):
    loss.backward()

for param in scaleModule_ext.parameters():
    print("batch gradient", param.grad)
    print("individual gradients", param.grad_batch)

print(
    "Does batch gradient match with individual gradients?",
    torch.allclose(
        scaleModule_ext.weight.grad, scaleModule_ext.weight.grad_batch.sum(axis=0)
    ),
)

# ensuring this test runs
assert torch.allclose(
    scaleModule_ext.weight.grad, scaleModule_ext.weight.grad_batch.sum(axis=0)
)

# %%
# Calculate the individual gradients with autograd and compare to BackPACK.
grad_batch_backpack = scaleModule_ext.weight.grad_batch
grad_batch_autograd = torch.zeros(grad_batch_backpack.shape)
for n in range(batch_size):
    scaleModule.zero_grad()
    loss = lossfunc(scaleModule(input[n].unsqueeze(0)), target[n].unsqueeze(0))
    loss.backward()
    grad_batch_autograd[n] = scaleModule.weight.grad

print("grad_batch_autograd", grad_batch_autograd)

print(
    "Does autograd and backpack individual gradients match?",
    torch.allclose(grad_batch_autograd, grad_batch_backpack),
)

# ensuring this test runs
assert torch.allclose(grad_batch_autograd, grad_batch_backpack)
