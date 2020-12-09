"""Extension hook example
==============================

The extension hook is a function called on each module after the BackPACK
extensions have run. It can be used to reduce memory overhead if the goal
is to compute transformations of BackPACK quantities. Information can be
compacted during a backward pass and obsolete tensors be freed manually.

Here, we use it to compute the Hessian trace after each module and free
the memory used to store the diagonal Hessian to reduce peak memory load.
"""

# %%
# Let's start by loading some dummy data and extending the model

import torch
from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential

from backpack import backpack, extend
from backpack.extensions import DiagHessian
from backpack.utils.examples import load_one_batch_mnist

# make deterministic
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data
X, y = load_one_batch_mnist(batch_size=128)
X, y = X.to(device), y.to(device)

# model
model = Sequential(Flatten(), Linear(784, 10)).to(device)
lossfunc = CrossEntropyLoss().to(device)

model = extend(model)
lossfunc = extend(lossfunc)

# %%
# Standard computation of the trace
# ---------------------------------

loss = lossfunc(model(X), y)

with backpack(DiagHessian()):
    loss.backward()

tr_after_backward = sum(param.diag_h.sum() for param in model.parameters())

print(f"Tr(H) after backward: {tr_after_backward:.3f} ")


# %%
# Let's clean up the computation graph and existing BackPACK buffers
del loss

for param in model.parameters():
    del param.diag_h


# %%
# Extension hook
# --------------

# %%
# The extension hook is a function that takes a ``torch.nn.Module`` (and returns
# ``None``). It is executed on each module after the BackPACK extensions have run.
#
# We use an object to store information from all modules. The hook will compute
# the trace of the Hessian for the block of parameters associated with the module
# and mark the tensors storing the diagonal Hessian to be freed.


class TraceHook:
    def __init__(self):
        """BackPACK extension hook that sums up the Hessian diagonal on the fly."""
        self.value = 0.0

    def sum_diag_h(self, module):
        """Sum ``value`` attribute with the diagonal Hessian elements."""
        for param in module.parameters():
            if hasattr(param, "diag_h"):
                self.value += param.diag_h.sum()
                delattr(param, "diag_h")


tr_hook = TraceHook()


# %%
# Hook computation of the trace
# ---------------------------------

loss = lossfunc(model(X), y)

with backpack(DiagHessian(), extension_hook=tr_hook.sum_diag_h):
    loss.backward()

tr_while_backward = tr_hook.value

print(f"Tr(H) while backward: {tr_while_backward:.3f}")
print(f"Same Tr(H)?           {torch.allclose(tr_after_backward, tr_while_backward)}")


# %%
# On memory usage
# -----------------
# The ``delattr`` and ``del`` functions do not directly free emory, but mark
# the tensor to be garbage collected by Python (as long as there are no other
# reference to the tensor.
#
# For the diagonal Hessian, the memory savings are rather small, as it has the
# same size as the gradient. For quantities that scale with batch and model size,
# like individual gradients the extension hook might make it possible to fit the
# computation in RAM where it would not be possible otherwise.
