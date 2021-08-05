"""Mini-batch sub-sampling
==========================

By default, BackPACK's extensions consider all samples in a mini-batch. Some extensions
support limiting the computations to a subset of samples. This example shows how to
restrict the computations to such a subset of samples.

This may be interesting for applications where parts of the samples are used for
different purposes, e.g. computing curvature and gradient information on different
subsets. Limiting the computations to fewer samples also reduces costs.

.. note::
   Not all extensions support sub-sampling yet. Please create a feature request in the
   repository if the extension you need is not supported.
"""

# %%
# Let's start by loading some dummy data and extending the model

from torch import allclose, cuda, device, manual_seed
from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential

from backpack import backpack, extend
from backpack.extensions import BatchGrad
from backpack.utils.examples import load_one_batch_mnist

# make deterministic
manual_seed(0)

dev = device("cuda" if cuda.is_available() else "cpu")

# data
X, y = load_one_batch_mnist(batch_size=128)
X, y = X.to(dev), y.to(dev)

# model
model = Sequential(Flatten(), Linear(784, 10)).to(dev)
lossfunc = CrossEntropyLoss().to(dev)

model = extend(model)
lossfunc = extend(lossfunc)

# %%
# Individual gradients for a mini-batch subset
# --------------------------------------------
#
# Let's say we only want to compute individual gradients for samples 0, 1,
# 13, and 42. Naively, we could perform the computation for all samples, then
# slice out the samples we care about.

# selected samples
subsampling = [0, 1, 13, 42]

loss = lossfunc(model(X), y)

with backpack(BatchGrad()):
    loss.backward()

# naive approach: compute for all, slice out relevant
naive = [p.grad_batch[subsampling] for p in model.parameters()]

# %%
# This is not efficient, as individual gradients are computed for all samples,
# most of them being discarded after. We can do better by specifying the active
# samples directly with the ``subsampling`` argument of
# :py:class:`BatchGrad <backpack.extensions.BatchGrad>`.

loss = lossfunc(model(X), y)

# efficient approach: specify active samples in backward pass
with backpack(BatchGrad(subsampling=subsampling)):
    loss.backward()

efficient = [p.grad_batch for p in model.parameters()]

# %%
# Let's verify that both ways yield the same result:

match = all(
    allclose(g_naive, g_efficient) for g_naive, g_efficient in zip(naive, efficient)
)

print(f"Naive and efficient sub-sampled individual gradients match? {match}")

if not match:
    raise ValueError("Naive and efficient sub-sampled individual gradient don't match.")
