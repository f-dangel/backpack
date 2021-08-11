"""Residual networks
====================
"""
# %%
# There are three different approaches to using BackPACK with ResNets.
#
# 1. :ref:`Custom ResNet`: (Only works for first-order extensions) Write your own model
#    by defining its forward pass. Trainable parameters must be in modules known to
#    BackPACK (e.g. :class:`torch.nn.Conv2d`, :class:`torch.nn.Linear`).
#
# 2. :ref:`Custom ResNet using BackPACK custom modules`: (Works for first- and second-
#    order extensions) Build your ResNet with custom modules provided by BackPACK
#    without overwriting the forward pass. This approach is useful if you want to
#    understand how BackPACK handles ResNets, or if you think building a container
#    module that implicitly defines the forward pass is more elegant than coding up
#    a forward pass.
#
# 3. :ref:`Any ResNet using BackPACK's converter`: (Works for first- and second-order
#    extensions) Convert your model into a BackPACK-compatible architecture.
#
# .. note::
#    ResNets are still an experimental feature. Always double-check your
#    results, as done in this example! Open an issue if you encounter a bug to help
#    us improve the support.
#
#    Not all extensions support ResNets (yet). Please create a feature request in the
#    repository if the extension you need is not supported.

# %%
# Let's get the imports out of the way.

from torch import allclose, cuda, device, rand, rand_like
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
x, y = load_one_batch_mnist(batch_size=32)
x, y = x.to(DEVICE), y.to(DEVICE)


# %%
# Custom ResNet
# -------------
# We can build a ResNet by extending :py:class:`torch.nn.Module`.
# As long as the layers with parameters (:py:class:`torch.nn.Conv2d`
# and :py:class:`torch.nn.Linear`) are ``nn`` modules, BackPACK can extend them,
# and this is all that is needed for first-order extensions.
# We can rewrite the :code:`forward` to implement the residual connection,
# and :py:func:`extend() <backpack.extend>` the resulting model.
#
# .. note::
#    Using in-place operations is not compatible with PyTorch's
#    :meth:`torch.nn.Module.register_full_backward_hook`. Therefore,
#    always use :code:`x = x + residual` instead of :code:`x += residual`.
class MyFirstResNet(Module):
    def __init__(self, C_in=1, C_hid=5, input_dim=(28, 28), output_dim=10):
        """Instantiate submodules that are used in the forward pass."""
        super().__init__()

        self.conv1 = Conv2d(C_in, C_hid, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(C_hid, C_hid, kernel_size=3, stride=1, padding=1)
        self.linear1 = Linear(input_dim[0] * input_dim[1] * C_hid, output_dim)
        if C_in == C_hid:
            self.shortcut = Identity()
        else:
            self.shortcut = Conv2d(C_in, C_hid, kernel_size=1, stride=1)

    def forward(self, x):
        """Manual implementation of the forward pass."""
        residual = self.shortcut(x)
        x = self.conv2(relu(self.conv1(x)))
        x = x + residual  # don't use: x += residual
        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        return x


model = extend(MyFirstResNet()).to(DEVICE)

# %%
# Using :py:class:`BatchGrad <backpack.extensions.BatchGrad>` in a
# :py:class:`with backpack(...) <backpack.backpack>` block,
# we can access the individual gradients for each sample.
#
# The loss does not need to be extended in this case either, as it does not
# have model parameters and BackPACK does not need to know about it for
# first-order extensions. This also means you can use any custom loss function.

loss = cross_entropy(model(x), y, reduction="sum")

with backpack(BatchGrad()):
    loss.backward()

for name, parameter in model.named_parameters():
    print(f"{name:>20}'s grad_batch shape: {parameter.grad_batch.shape}")

# %%
# To check that everything works, let's compute one individual gradient with
# PyTorch (using a single sample in a forward and backward pass)
# and compare it with the one computed by BackPACK.

sample_to_check = 1
x_to_check = x[[sample_to_check]]
y_to_check = y[[sample_to_check]]

model.zero_grad()
loss = cross_entropy(model(x_to_check), y_to_check)
loss.backward()

print("Do the individual gradients match?")
for name, parameter in model.named_parameters():
    match = allclose(parameter.grad_batch[sample_to_check], parameter.grad, atol=1e-7)
    print(f"{name:>20}: {match}")
    if not match:
        raise AssertionError("Individual gradients don't match!")

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
