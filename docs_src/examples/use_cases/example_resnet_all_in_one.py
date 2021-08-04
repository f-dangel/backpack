"""Residual networks
====================
"""
# %%
# There are three different approaches to using ResNets.
#
# 1. Custom ResNet: This approach is useful if you need only first order
# information and your ResNet is very simple.
#
# 2. Custom ResNet using BackPACK custom modules: This approach is useful if you
# need second order information and your ResNet is very simple.
#
# 3. Any ResNet using BackPACK's converter: This uses BackPACK's convenient
# converter function. It is suitable for complex architectures.

# %%
# Let's get the imports out of the way.
from torch import cuda, device, rand, rand_like
from torch.nn import (
    Conv2d,
    CrossEntropyLoss,
    Flatten,
    Identity,
    Linear,
    Module,
    MSELoss,
    ReLU,
    Sequential,
)
from torch.nn.functional import cross_entropy, relu
from torchvision.models import resnet18

from backpack import backpack, extend
from backpack.custom_module.branching import ActiveIdentity, Parallel, SumModule
from backpack.custom_module.graph_utils import print_table
from backpack.extensions import BatchGrad, DiagGGNExact
from backpack.utils.examples import load_one_batch_mnist

DEVICE = device("cuda:0" if cuda.is_available() else "cpu")


# %%
# Custom ResNet
# -------------
# Let's define a basic ResNet.
#
# Note that using any in-place operations is not compatible with PyTorch's
# `register_full_backward_hook`.
# Therefore, always use `x = x + residual` instead of `x += residual`.
class MyFirstResNet(Module):
    def __init__(self, C_in=1, C_hid=5, input_dim=(28, 28), output_dim=10):
        super().__init__()

        self.conv1 = Conv2d(C_in, C_hid, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(C_hid, C_hid, kernel_size=3, stride=1, padding=1)
        self.linear1 = Linear(input_dim[0] * input_dim[1] * C_hid, output_dim)
        if C_in == C_hid:
            self.shortcut = Identity()
        else:
            self.shortcut = Conv2d(C_in, C_hid, kernel_size=1, stride=1)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv2(relu(self.conv1(x)))
        x = x + residual  # don't use: x += residual
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x


x, y = load_one_batch_mnist(batch_size=32)
x, y = x.to(DEVICE), y.to(DEVICE)
model = extend(MyFirstResNet()).to(DEVICE)
loss = cross_entropy(model(x), y, reduction="sum")
with backpack(BatchGrad()):
    loss.backward()
for name, parameter in model.named_parameters():
    print(f"{name}'s grad_batch: {parameter.grad_batch.shape}")

# %%
# Custom ResNet using BackPACK custom modules
# -------------
# For second order extensions, every single node in the computation graph needs
# to be extended by BackPACK.
# This is why BackPACK offers custom modules to be used in ResNets:
#
# :py:class:`Parallel<backpack.branching.Parallel>` is similar to
# :py:class:`torch.nn.Sequential` being a container for a sequence of modules.
#
# :py:class:`SumModule<backpack.branching.SumModule>`, which is internally used as the default
# aggregation function in :py:class:`Parallel<backpack.branching.Parallel>`.
#
# :py:class:`ActiveIdentity<backpack.branching.ActiveIdentity>`, which acts like
# PyTorch's identity, but fixes the backward hook execution order by inserting a new
# node into the graph during a forward pass.
# For PyTorch >= 1.9.0 it is possible to use :py:class:`Identity` instead.
C_in = 1
C_hid = 2
input_dim = (28, 28)
output_dim = 10

model = extend(
    Sequential(
        Conv2d(C_in, C_hid, kernel_size=3, stride=1, padding=1),
        ReLU(),
        Parallel(
            ActiveIdentity(),
            Sequential(
                Conv2d(C_hid, C_hid, kernel_size=3, stride=1, padding=1),
                ReLU(),
            ),
            merge_module=SumModule(),
        ),
        Flatten(),
        Linear(input_dim[0] * input_dim[1] * C_hid, output_dim),
    )
).to(DEVICE)
loss_function = extend(CrossEntropyLoss(reduction="mean")).to(DEVICE)
loss = loss_function(model(x), y)
with backpack(DiagGGNExact()):
    loss.backward()
for name, parameter in model.named_parameters():
    print(f"{name}'s diag_ggn_exact: {parameter.diag_ggn_exact.shape}")

# %%
# Any ResNet using BackPACK's converter
# -------------
# For more complex architectures, e.g. resnet18 from torchvision,
# BackPACK has a converter function.
# It creates a torch.fx.GraphModule and converts all known schemes to a BackPACK
# compatible module.
# Whether the graph consists exclusively of modules can be checked in the table.
#
# Resnet18 has to be in evaluation mode, because there are BatchNorm layers involved.
# For these, individual gradients can be computed but are not well-defined.
#
# Here, we use a limited number of classes, because the DiagGGN extension memory usage
# scales with it. In theory, it should work with more classes, but the current
# implementation is not memory efficient enough.
model = resnet18(num_classes=5).to(DEVICE).eval()
loss_function = extend(MSELoss())
model = extend(model, use_converter=True)
print_table(model)

# %%
# If successful, first and second order quantities can be computed.
inputs = rand(4, 3, 7, 7, device=DEVICE)  # (128, 3, 224, 224)
output = model(inputs)
loss = loss_function(output, rand_like(output))

with backpack(DiagGGNExact()):
    loss.backward()
for name, parameter in model.named_parameters():
    print(f"{name}'s diag_ggn_exact: {parameter.diag_ggn_exact.shape}")
