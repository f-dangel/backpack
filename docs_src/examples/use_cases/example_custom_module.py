"""Custom module example.

This tutorial shows how to support a custom module in a simple fashion.
"""

# %%
#
import torch
from torch.autograd import Function
from torch.nn import CrossEntropyLoss

from backpack import backpack, extend
from backpack.extensions import BatchGrad
from backpack.extensions.firstorder.base import FirstOrderModuleExtension

# make deterministic
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# Define custom function
class ScaleFunction(Function):
    """Function that scales the function with weight."""

    @staticmethod
    def forward(ctx, input, weight):
        """Forward pass.

        Args:
            ctx(Any): context object
            input(torch.Tensor): input
            weight(torch.Tensor): weight

        Returns:
            torch.Tensor: input * weight
        """
        ctx.save_for_backward(input, weight)
        output = input * weight
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Calculates derivatives wrt to input and weight.

        Args:
            ctx(Any): context object
            grad_output(torch.Tensor): output gradient

        Returns:
            tuple[torch.Tensor, torch.Tensor]: grad_input, grad_weight
        """
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output * weight
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output * input

        return grad_input, grad_weight


# %%
# Define custom module
class ScaleModule(torch.nn.Module):
    """Defines the module based on ScaleFunction."""

    def __init__(self, input_size=(1,), weight=2.0):
        """Initializes scale module.

        Args:
            input_size: size of input is equivalent to output size
            weight(float): initializes whole weight vector with weight.
                Defaults to 2.0.
        """
        super(ScaleModule, self).__init__()
        self.input_size = input_size

        self.weight = torch.nn.Parameter(torch.full(input_size, weight))

    def forward(self, input):
        """Defines forward pass based on ScaleFunction.

        Args:
            input(torch.Tensor): input

        Returns:
            torch.Tensor: Result from ScaleFunction
        """
        return ScaleFunction.apply(input, self.weight)


# %%
# Create random data
batch_size = 10
input_size = 4
input = torch.randn(batch_size, input_size)
input.requires_grad = True
target = torch.randint(0, 2, (batch_size,))

# %%
# Test custom module
scaleModule = ScaleModule(input_size=(input_size,))
lossfunc = CrossEntropyLoss()
loss = lossfunc(scaleModule(input), target)

loss.backward()

for param in scaleModule.parameters():
    print("batch gradient", param.grad)


# %%
# code
class ScaleModuleBatchGrad(FirstOrderModuleExtension):
    """Extract indiviual gradients for ``ScaleModule``."""

    def __init__(self):
        """Initializes scale module batch extension."""
        super().__init__(params=["weight"])

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        """Extract individual gradients for ``ScaleModule``'s ``weight`` parameter.

        Args:
            ext(BatchGrad): extension that is used
            module(ScaleModule): module that performed forward pass
            g_inp(tuple[torch.Tensor]): input gradient tensors
            g_out(tuple[torch.Tensor]): output gradient tensors
            bpQuantities(None): additional quantities for second order

        Returns:
            torch.Tensor: individual gradients
        """
        # output is saved under field output
        # print("module.output", module.output)
        # input i is saved under field input[i]
        # print("module.input0", module.input0)
        # gradient of output
        # print("g_out[0]", g_out[0])
        print(type(g_inp[0]))
        return g_out[0] * module.input0


# %%
# backward with backpack
scaleModule = extend(scaleModule)
lossfunc = extend(lossfunc)
scaleModule.zero_grad()
loss = lossfunc(scaleModule(input), target)
print("loss", loss)

ext = BatchGrad()
ext.add_module_extension(ScaleModule, ScaleModuleBatchGrad())
with backpack(ext):
    loss.backward()

for param in scaleModule.parameters():
    print("batch gradient", param.grad)
    print("individual gradients", param.grad_batch)

print(
    "Does batch gradient match with individual gradients?",
    torch.allclose(scaleModule.weight.grad, scaleModule.weight.grad_batch.sum(axis=0)),
)
