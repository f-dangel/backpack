"""Automatic support
====================

This tutorial explains how to support new layers in BackPACK without knowledge
about derivatives or autodiff internals.

This is possible through a new, and experimental, :class:`AutomaticDerivatives`
class, which uses PyTorch's :mod:`torch.autograd` under the hood.
It makes it easy to quickly support new layers. However, this comes at the cost
of performance, because the autograd-based solution simply cannot avoid internal
re-computation and for loops.

If you want to support a new layer efficiently, please check out the
:ref:`Custom module example`.

The automatic support we describe in this tutorial works as follows:

1. Define a derivative class for your layer. All you have to do is specify the forward
   pass. The derivatives required by BackPACK will be derived from that.

2. Define a module extension for the BackPACK extension you wish to compute, and
   feed the above derivatives into it.

3. Register the mapping between module and module extension.

We will demonstrate these steps for the group normalization layer
(:class:`torch.nn.GroupNorm`), which currently has no efficient support in
BackPACK.

Let's get the imports out of our way.

"""  # noqa: B950

from typing import Optional

from torch import Tensor, cuda, device, manual_seed, rand, zeros
from torch.autograd import grad
from torch.nn import (
    Conv2d,
    Flatten,
    GroupNorm,
    Linear,
    MSELoss,
    Parameter,
    ReLU,
    Sequential,
    Sigmoid,
)
from torch.nn.functional import group_norm
from torch.nn.utils.convert_parameters import parameters_to_vector

from backpack import backpack, extend, extensions
from backpack.core.derivatives.automatic import AutomaticDerivatives, ForwardCallable
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule
from backpack.hessianfree.ggnvp import ggn_vector_product
from backpack.utils.convert_parameters import vector_to_parameter_list

# make deterministic
manual_seed(0)

dev = device("cuda" if cuda.is_available() else "cpu")

# %%
#
# Define a derivative class
# -------------------------
#
# The heavy lifting inside BackPACK is abstracted into a class that implements
# all kinds of derivatives. BackPACK's core provides a class called
# :class:`AutomaticDerivatives` that can be used to support new layers without
# implementing their derivatives. The derivatives are simply implemented using
# :mod:`torch.autograd`. This is less efficient than hand-crafted derivatives, but
# requires less human time and autodiff expertise.
#
#
# To create a new derivatives class, inherit from :class:`AutomaticDerivatives`
# and implement its abstract method :func:`as_functional` which returns a
# function that performs the layer's forward pass.
#
# Let's create such a class for the group normalization layer:


class GroupNormAutomaticDerivatives(AutomaticDerivatives):
    """Automatic derivatives for ``torch.nn.GroupNorm``."""

    @staticmethod
    def as_functional(module: GroupNorm) -> ForwardCallable:
        """Return the ``GroupNorm`` layer's forward pass function.

        Args:
            module: The ``GroupNorm`` layer whose forward pass function is returned.

        Returns:
            The ``GroupNorm`` layer's forward pass function which consumes the layer
            input and parameters and produces the output.
        """

        def forward(
            x: Tensor, weight: Optional[Parameter], bias: Optional[Parameter]
        ) -> Tensor:
            """Map layer input and parameters to layer output."""
            return group_norm(x, module.num_groups, weight, bias, module.eps)

        return forward


# %%
#
# Define a module extension
# -------------------------
#
# Module extensions in BackPACK define what computations are carried out for a specific
# layer and extensions.
#
# Let's support per-datum gradients (i.e. BackPACK's :class:`BatchGrad
# <backpack.extensions.BatchGrad>` extension) for the group normalization layer. To that,
# we have to define a module extension that uses the derivatives class we just created:


class BatchGradGroupNormAutomatic(BatchGradBase):
    """BatchGrad extension for ``torch.nn.GroupNorm`` using automatic derivatives."""

    def __init__(self):
        """Initialize the extension."""
        super().__init__(
            derivatives=GroupNormAutomaticDerivatives(),
            # ``params`` only needs to be specified if the layer has learnable params
            params=["weight", "bias"],
        )


# %%
#
# Register the module extension
# -----------------------------
#
# To tell BackPACK to execute the above module extension when it encounters a group
# normalization layer during a backward pass with the
# :class:`BatchGrad <backpack.extensions.BatchGrad` extension, we need to register the
# mapping every time we create an extension instance as follows:

ext = extensions.BatchGrad()
ext.set_module_extension(GroupNorm, BatchGradGroupNormAutomatic())

# %%
# Verify correctness
# ------------------
#
# It is important to verify that the new module extension works as expected.
#
# Let's create a synthetic problem and make sure the per-datum gradients computed with
# BackPACK match those computed with PyTorch's autograd.

BATCH_SIZE = 10
X = rand(BATCH_SIZE, 3, 28, 28, device=dev)
y = rand(BATCH_SIZE, 4, device=dev)

model = Sequential(
    Conv2d(3, 4, 5, stride=2),
    ReLU(),
    GroupNorm(2, 4),
    Conv2d(4, 2, 3),
    Sigmoid(),
    Flatten(),
    Linear(200, 4),
).to(dev)
lossfunc = MSELoss(reduction="mean").to(dev)

# %%
#
# First, we compute the ground truth: the per-datum gradients with PyTorch's autograd.

params = [p for p in model.parameters() if p.requires_grad]
grad_batch_true = [
    zeros(BATCH_SIZE, *p.shape, device=p.device, dtype=p.dtype) for p in params
]

for n, (X_n, y_n) in enumerate(zip(X.split(1), y.split(1))):
    loss = lossfunc(model(X_n), y_n)
    for g, g_n in zip(grad_batch_true, grad(loss, params)):
        g[n] = g_n / BATCH_SIZE

# %%
#
# Now, we will compute the per-datum gradients with BackPACK.
#
# As usual, we have to :func:`extend <backpack.extend>` the model and loss function,
# then call the :class:`with backpack(...) <backpack.backpack>` context manager with
# the new extension, and finally compare the computation's results.

model = extend(model)
lossfunc = extend(lossfunc)

# Remember that we registered the mapping between the group normalization layer and
# our module extension in ``ext`` earlier
with backpack(ext):
    loss = lossfunc(model(X), y)
    loss.backward()

grad_batch = [p.grad_batch for p in params]

# compare
if len(grad_batch) != len(grad_batch_true):
    raise AssertionError("Parameter list structure does not match.")

if not all(g.allclose(g_true) for g, g_true in zip(grad_batch, grad_batch_true)):
    raise AssertionError("Per-datum gradients do not match.")
else:
    print("Per-datum gradients match.")

# %%
#
# It works!
#
# We can now compute per-datum gradients for the parameters of a group norm layer.

# %%
#
# Repeat for other extensions
# ---------------------------
#
# So far, we demonstrated everything for one extension. Other extensions follow the same
# process.
#
# We illustrate this here for the exact GGN diagonal
# (:class:`DiagGGNExact <backpack.extensions.DiagGGNExact>`).
#
# Let's re-create the synthetic data and model to avoid side effects from before.

BATCH_SIZE = 10
X = rand(BATCH_SIZE, 3, 28, 28, device=dev)
y = rand(BATCH_SIZE, 4, device=dev)

model = Sequential(
    Conv2d(3, 4, 5, stride=3),
    ReLU(),
    GroupNorm(2, 4),
    Conv2d(4, 2, 3, stride=2),
    Sigmoid(),
    Flatten(),
    Linear(18, 4),
).to(dev)
lossfunc = MSELoss()

# %%
#
# First, let's compute our ground truth using PyTorch's autodiff.

params = [p for p in model.parameters() if p.requires_grad]
ggn_dim = sum(p.numel() for p in params)
diag_ggn_flat = zeros(ggn_dim, device=X.device, dtype=X.dtype)

outputs = model(X)
loss = lossfunc(outputs, y)

# compute GGN-vector products with all one-hot vectors
for d in range(ggn_dim):
    # create unit vector d
    e_d = zeros(ggn_dim, device=X.device, dtype=X.dtype)
    e_d[d] = 1.0
    # convert to list format
    e_d = vector_to_parameter_list(e_d, params)

    # multiply GGN onto the unit vector -> get back column d of the GGN
    ggn_e_d = ggn_vector_product(loss, outputs, model, e_d)
    # flatten
    ggn_e_d = parameters_to_vector(ggn_e_d)

    # extract the d-th entry (which is on the GGN's diagonal)
    diag_ggn_flat[d] = ggn_e_d[d]

print(f"Tr(GGN, autograd): {diag_ggn_flat.sum():.3f}")

# %%
#
# Next, let's define a module extension that tells
# :class:`DiagGGNExact <backpack.extensions.DiagGGNExact>` what to do if it encounters
# a group normalization layer.


class DiagGGNExactGroupNormAutomatic(DiagGGNBaseModule):
    """GGN diagonal computation for ``torch.nn.GroupNorm`` via automatic derivatives."""

    def __init__(self):
        """Set up the derivatives."""
        super().__init__(
            GroupNormAutomaticDerivatives(), params=["weight", "bias"], sum_batch=True
        )


# %%
#
# Register the mapping ...

ext = extensions.DiagGGNExact()
ext.set_module_extension(GroupNorm, DiagGGNExactGroupNormAutomatic())

# %%
#
# ... and run a backward pass with BackPACK

model = extend(model)
lossfunc = extend(lossfunc)

with backpack(ext):
    loss = lossfunc(model(X), y)
    loss.backward()

# %%
#
# Finally, let's collect the result and compare it with autograd.

diag_ggn_flat_backpack = parameters_to_vector([p.diag_ggn_exact for p in params])
print(f"Tr(GGN, BackPACK): {diag_ggn_flat_backpack.sum():.3f}")

if not diag_ggn_flat.allclose(diag_ggn_flat_backpack):
    raise AssertionError("Exact GGN diagonals do not match.")
else:
    print("Exact GGN diagonals match.")

# %%
#
# Works as well, great!
