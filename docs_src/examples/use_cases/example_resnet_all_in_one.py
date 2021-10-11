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
# 2. :ref:`Custom ResNet with BackPACK custom modules`: (Works for first- and second-
#    order extensions) Build your ResNet with custom modules provided by BackPACK
#    without overwriting the forward pass. This approach is useful if you want to
#    understand how BackPACK handles ResNets, or if you think building a container
#    module that implicitly defines the forward pass is more elegant than coding up
#    a forward pass.
#
# 3. :ref:`Any ResNet with BackPACK's converter`: (Works for first- and second-order
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

from torch import (
    allclose,
    cat,
    cuda,
    device,
    int32,
    linspace,
    manual_seed,
    rand,
    rand_like,
)
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
from backpack.custom_module.branching import Parallel, SumModule
from backpack.custom_module.graph_utils import BackpackTracer
from backpack.extensions import BatchGrad, DiagGGNExact
from backpack.utils.examples import autograd_diag_ggn_exact, load_one_batch_mnist

manual_seed(0)
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
# The loss does not need to be extended in this case either, as it does not
# have model parameters and BackPACK does not need to know about it for
# first-order extensions. This also means you can use any custom loss function.
#
# Using :py:class:`BatchGrad <backpack.extensions.BatchGrad>` in a
# :py:class:`with backpack(...) <backpack.backpack>` block,
# we can access the individual gradients for each sample.

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
    match = allclose(parameter.grad_batch[sample_to_check], parameter.grad, atol=1e-6)
    print(f"{name:>20}: {match}")
    if not match:
        raise AssertionError("Individual gradients don't match!")

# %%
# Custom ResNet with BackPACK custom modules
# -------------
# Second-order extensions only work if every node in the computation graph is an
# ``nn`` module that can be extended by BackPACK. The above ResNet class
# :py:class:`MyFirstResNet<MyFirstResNet>` does not satisfy these conditions, because
# it implements the skip connection via :py:func:`torch.add` while overwriting the
# :py:meth:`forward() <torch.nn.Module.forward>` method.
#
# To build ResNets without overwriting the forward pass, BackPACK offers custom modules:
#
# 1. :py:class:`Parallel<backpack.branching.Parallel>` is similar to
#    :py:class:`torch.nn.Sequential`, but implements a container for a parallel sequence
#    of modules (followed by an aggregation module), rather than a sequential one.
#
# 2. :py:class:`SumModule<backpack.branching.SumModule>` is the module that takes the
#    role of :py:func:`torch.add` in the previous example. It sums up multiple inputs.
#    We will use it to merge the skip connection.
#
# With the above modules, we can build a simple ResNet as a container that implicitly
# defines the forward pass:

C_in = 1
C_hid = 2
input_dim = (28, 28)
output_dim = 10

model = Sequential(
    Conv2d(C_in, C_hid, kernel_size=3, stride=1, padding=1),
    ReLU(),
    Parallel(  # skip connection with ReLU-activated convolution
        Identity(),
        Sequential(
            Conv2d(C_hid, C_hid, kernel_size=3, stride=1, padding=1),
            ReLU(),
        ),
        merge_module=SumModule(),
    ),
    Flatten(),
    Linear(input_dim[0] * input_dim[1] * C_hid, output_dim),
)

model = extend(model.to(DEVICE))
loss_function = extend(CrossEntropyLoss(reduction="mean")).to(DEVICE)


# %%
# This ResNets supports BackPACK's second-order extensions:

loss = loss_function(model(x), y)

with backpack(DiagGGNExact()):
    loss.backward()

for name, parameter in model.named_parameters():
    print(f"{name}'s diag_ggn_exact: {parameter.diag_ggn_exact.shape}")

diag_ggn_exact_vector = cat([p.diag_ggn_exact.flatten() for p in model.parameters()])

# %%
# Comparison with :py:mod:`torch.autograd`:
#
# .. note::
#
#    Computing the full GGN diagonal with PyTorch's built-in automatic differentiation
#    can be slow, depending on the number of parameters. To reduce run time, we only
#    compare some elements of the diagonal.

num_params = sum(p.numel() for p in model.parameters())
num_to_compare = 10
idx_to_compare = linspace(0, num_params - 1, num_to_compare, device=DEVICE, dtype=int32)

diag_ggn_exact_to_compare = autograd_diag_ggn_exact(
    x, y, model, loss_function, idx=idx_to_compare
)

print("Do the exact GGN diagonals match?")
for idx, element in zip(idx_to_compare, diag_ggn_exact_to_compare):
    match = allclose(element, diag_ggn_exact_vector[idx], atol=1e-6)
    print(f"Diagonal entry {idx:>6}: {match}")
    if not match:
        raise AssertionError("Exact GGN diagonals don't match!")

# %%
# Any ResNet with BackPACK's converter
# -------------
# If you are not building a ResNet through custom modules but for instance want to
# use a prominent ResNet from :py:mod:`torchvision.models`, BackPACK offers a converter.
# It analyzes the model and tries to turn it into a compatible architecture. The result
# is a :py:class:`torch.fx.GraphModule` that exclusively consists of modules.
#
# Here, we demo the converter on :py:class:`resnet18 <torchvision.models.resnet18>`.
#
# .. note::
#
#    :py:class:`resnet18 <torchvision.models.resnet18>` has to be in evaluation mode,
#    because it contains batch normalization layers that are not supported in train
#    mode by the second-order extension used in this example.
#
# Let's create the model, and convert it in the call to :py:func:`extend <backpack.extend>`:

loss_function = extend(MSELoss().to(DEVICE))
model = resnet18(num_classes=5).to(DEVICE).eval()

# use BackPACK's converter to extend the model (turned off by default)
model = extend(model, use_converter=True)

# %%
# To get an understanding what happened, we can inspect the model's graph with the
# following helper function:


def print_table(module: Module) -> None:
    """Prints a table of the module.

    Args:
        module: module to analyze
    """
    graph = BackpackTracer().trace(module)
    graph.print_tabular()


print_table(model)

# %%
# Admittedly, the converted :py:class:`resnet18 <torchvision.models.resnet18>`'s graph
# is quite large. Note however that it fully consists of modules (indicated by
# ``call_module`` in the first table column) such that BackPACK's hooks can
# successfully backpropagate additional information for its second-order extensions
# (first-order extensions work, too).
#
# Let's verify that second-order extensions are working:

x = rand(4, 3, 7, 7, device=DEVICE)  # (128, 3, 224, 224)
output = model(x)
y = rand_like(output)

loss = loss_function(output, y)

with backpack(DiagGGNExact()):
    loss.backward()

for name, parameter in model.named_parameters():
    print(f"{name}'s diag_ggn_exact: {parameter.diag_ggn_exact.shape}")

diag_ggn_exact_vector = cat([p.diag_ggn_exact.flatten() for p in model.parameters()])

# %%
# Comparison with :py:mod:`torch.autograd`:
#
# .. note::
#
#    Computing the full GGN diagonal with PyTorch's built-in automatic differentiation
#    can be slow, depending on the number of parameters. To reduce run time, we only
#    compare some elements of the diagonal.

num_params = sum(p.numel() for p in model.parameters())
num_to_compare = 10
idx_to_compare = linspace(0, num_params - 1, num_to_compare, device=DEVICE, dtype=int32)

diag_ggn_exact_to_compare = autograd_diag_ggn_exact(
    x, y, model, loss_function, idx=idx_to_compare
)

print("Do the exact GGN diagonals match?")
for idx, element in zip(idx_to_compare, diag_ggn_exact_to_compare):
    match = allclose(element, diag_ggn_exact_vector[idx], atol=1e-6)
    print(f"Diagonal entry {idx:>8}: {match}")
    if not match:
        raise AssertionError("Exact GGN diagonals don't match!")
