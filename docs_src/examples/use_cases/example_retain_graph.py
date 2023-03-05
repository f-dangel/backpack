r"""BackPACK's retain_graph option
==================================

This tutorial demonstrates how to perform multiple backward passes through the
same computation graph with BackPACK. This option can be useful if you run into
out-of-memory errors. If your computation can be chunked, you might consider
distributing it onto multiple backward passes to reduce peak memory.

Our use case for such a quantity is the GGN diagonal of an auto-encoder's
reconstruction error.

But first, the imports:
"""

from functools import partial
from time import time
from typing import List

from memory_profiler import memory_usage
from torch import Tensor, allclose, manual_seed, rand, zeros_like
from torch.nn import Conv2d, ConvTranspose2d, Flatten, MSELoss, Sequential, Sigmoid

from backpack import backpack, extend
from backpack.custom_module.slicing import Slicing
from backpack.extensions import DiagGGNExact

# make deterministic
manual_seed(0)

# %%
#
# Setup
# -----
#
# Let :math:`f_{\mathbf{\theta}}` denote the auto-encoder, and
# :math:`\mathbf{x'} = f_{\mathbf{\theta}}(\mathbf{x}) \in \mathbb{R}^M` its
# reconstruction of an input :math:`\mathbf{x} \in \mathbb{R}^M`. The
# associated reconstruction error is measured by the mean squared error
#
# .. math::
#     \ell(\mathbf{\theta})
#     =
#     \frac{1}{M}
#     \left\lVert f_{\mathbf{\theta}}(\mathbf{x}) - \mathbf{x} \right\rVert^2_2\,.
#
# On a batch of :math:`N` examples, :math:`\mathbf{x}_1, \dots, \mathbf{x}_N`,
# the loss is
#
# .. math::
#     \mathcal{L}(\mathbf{\theta})
#     =
#     \frac{1}{N} \frac{1}{M}
#     \sum_{n=1}^N
#     \left\lVert f_{\mathbf{\theta}}(\mathbf{x}_n) - \mathbf{x}_n \right\rVert^2_2\,.
#
# Let's create a toy model and data:

# data
batch_size, channels, spatial_dims = 5, 3, (32, 32)
X = rand(batch_size, channels, *spatial_dims)

# model (auto-encoder)
hidden_channels = 10

encoder = Sequential(
    Conv2d(channels, hidden_channels, 3),
    Sigmoid(),
)
decoder = Sequential(
    ConvTranspose2d(hidden_channels, channels, 3),
    Flatten(),
)
model = Sequential(
    encoder,
    decoder,
)
loss_func = MSELoss()

# %%
#
# We will use BackPACK to compute the GGN diagonal of the mini-batch loss. To
# do that, we need to :py:func:`extend <backpack.extend>` model and loss
# function.

model = extend(model)
loss_func = extend(loss_func)

# %%
#
# GGN diagonal in one backward pass
# ---------------------------------
#
# As usual, we can compute the GGN diagonal for the mini-batch loss in a single
# backward pass. The following function does that:


def diag_ggn_one_pass() -> List[Tensor]:
    """Compute the GGN diagonal in a single backward pass.

    Returns:
        GGN diagonal in parameter list format.
    """
    reconstruction = model(X)
    error = loss_func(reconstruction, X.flatten(start_dim=1))

    with backpack(DiagGGNExact()):
        error.backward()

    return [p.diag_ggn_exact.clone() for p in model.parameters() if p.requires_grad]


# %%
#
# Let's run it and determine (i) its peak memory consumption and (ii) its run
# time.

print("GGN diagonal in one backward pass:")
start = time()
max_mem, diag_ggn = memory_usage(
    diag_ggn_one_pass, interval=1e-3, max_usage=True, retval=True
)
end = time()

print(f"\tPeak memory [MiB]: {max_mem:.2e}")
print(f"\tTime [s]: {end-start:.2e}")

# %%
#
# The memory consumption is pretty high, although our model is relatively
# small! If we make the model deeper, or increase the mini-batch size, we will
# quickly run out of memory.
#
# This is because computing the GGN diagonal scales with the network's output
# dimension. For classification settings like MNIST and CIFAR-10, this number
# is relatively small (:code:`10`). But for an auto-encoder, this number is the
# input dimension :code:`M`, which in our case is

print(f"Output dimension: {model(X).shape[1:].numel()}")

# %%
#
# We will now take a look at how to circumvent the high peak memory by
# distributing the computation over multiple backward passes.

# %%
#
# GGN diagonal in chunks
# ----------------------
#
# The GGN diagonal computation can be distributed across multiple backward
# passes. This greatly reduces peak memory consumption.
#
# To see this, let's consider the GGN diagonal for a single example
# :math:`\mathbf{x}`,
#
# .. math::
#     \mathrm{diag}
#     \left(
#     \left[
#     \mathbf{J}_{\mathbf{\theta}} f_{\mathbf{\theta}}(\mathbf{x})
#     \right]^\top
#     \left[
#     \frac{2}{M} \mathbf{I}_{M\times M}
#     \right]
#     \mathbf{J}_{\mathbf{\theta}} f_{\mathbf{\theta}}(\mathbf{x})
#     \right)\,,
#
# with the :math:`M \times |\mathbf{\theta}|` Jacobian
# :math:`\mathbf{J}_{\mathbf{\theta}} f_{\mathbf{\theta}}(\mathbf{x})` of the
# model, and :math:`\frac{2}{M} \mathbf{I}_{M\times M}` the mean squared
# error's Hessian w.r.t. the reconstructed input. Here you can see that the
# memory consumption scales with the output dimension, as we need to compute
# :code:`M` vector-Jacobian products.
#
# Let :math:`S`, the chunk size, be a number that divides the output dimension
# :math:`M`. Then, we can decompose the above computation into chunks:
#
# .. math::
#     \frac{S}{M}
#     \left\{
#     \mathrm{diag}
#     \left(
#     \left[
#     \mathbf{J}_{\mathbf{\theta}} f_{\mathbf{\theta}}(\mathbf{x})
#     \right]^\top_{:, 0:S}
#     \left[
#     \frac{2}{S} \mathbf{I}_{S\times S}
#     \right]
#     \left[
#     \mathbf{J}_{\mathbf{\theta}} f_{\mathbf{\theta}}(\mathbf{x})
#     \right]_{0:S, :}
#     \right) \right.
#     \\
#     +
#     \left.
#     \mathrm{diag}
#     \left(
#     \left[
#     \mathbf{J}_{\mathbf{\theta}} f_{\mathbf{\theta}}(\mathbf{x})
#     \right]^\top_{:, S: 2S}
#     \left[
#     \frac{2}{S} \mathbf{I}_{S\times S}
#     \right]
#     \left[
#     \mathbf{J}_{\mathbf{\theta}} f_{\mathbf{\theta}}(\mathbf{x})
#     \right]_{:, S:2S}
#     \right)
#     \right.
#     \\
#     +
#     \left.
#     \mathrm{diag}
#     \left(
#     \left[
#     \mathbf{J}_{\mathbf{\theta}} f_{\mathbf{\theta}}(\mathbf{x})
#     \right]^\top_{:, 2S: 3S}
#     \left[
#     \frac{2}{S} \mathbf{I}_{S\times S}
#     \right]
#     \left[
#     \mathbf{J}_{\mathbf{\theta}} f_{\mathbf{\theta}}(\mathbf{x})
#     \right]_{:, 2S: 3S}
#     \right)
#     +
#     \dots
#     \right\}\,.
#
# Each summand is the GGN diagonal of the mean squared error on a chunk
#
# .. math::
#     \tilde{\ell}(\mathbf{\theta})
#     =
#     \frac{1}{S}
#     \lVert
#     [f_{\mathbf{\theta}}(\mathbf{x})]_{i S: (i+1) S}
#     -
#     [\mathbf{x}]_{i S: (i+1) S}
#     \rVert_2^2\,,
#     \qquad i = 0, 1, \dots, \frac{M}{S} - 1\,,
#
# and its memory consumption scales with :math:`S < M`.
#
# In summary, the computation split works as follows:
#
# - Compute :math:`f_{\mathbf{\theta}}(\mathbf{x})` in a single forward pass.
#
# - Compute the reconstruction error for a chunk and its GGN in one backward
#   pass.
#
# - Repeat the last step for the other chunks. Accumulate the GGN diagonals
#   over all chunks.
#
# (This carries over to the mini-batch case in a straightforward fashion. We
# avoid the presentation here because of the involved notation, though.)
#
# Note that because we perform multiple backward passes, we need to tell
# PyTorch (and BackPACK) to retain the graph.
#
# To slice out a chunk, we use BackPACK's :py:class:`Slicing
# <backpack.custom_module.slicing>` module.
#
# Here is the implementation:


def diag_ggn_multiple_passes(num_chunks: int) -> List[Tensor]:
    """Compute the GGN diagonal in multiple backward passes.

    Uses less memory than ``diag_ggn_one_pass`` if ``num_chunks > 1``.
    Does the same as ``diag_ggn_one_pass`` for ``num_chunks = 1``.

    Args:
        num_chunks: Number of backward passes. Must divide the model's output dimension.

    Returns:
        GGN diagonal in parameter list format.

    Raises:
        ValueError:
            If ``num_chunks`` does not divide the model's output dimension.
        NotImplementedError:
            If the model does not return a batched vector (the slicing logic is only
            implemented for batched vectors, i.e. 2d tensors).
    """
    reconstruction = model(X)

    if reconstruction.numel() % num_chunks != 0:
        raise ValueError("Network output must be divisible by number of chunks.")
    if reconstruction.dim() != 2:
        raise NotImplementedError("Slicing logic only implemented for 2d outputs.")

    chunk_size = reconstruction.shape[1:].numel() // num_chunks
    diag_ggn_exact = [zeros_like(p) for p in model.parameters()]

    for idx in range(num_chunks):
        # set up the layer that extracts the current slice
        slicing = (slice(None), slice(idx * chunk_size, (idx + 1) * chunk_size))
        chunk_module = extend(Slicing(slicing))

        # compute the chunk's loss
        sliced_reconstruction = chunk_module(reconstruction)
        sliced_X = X.flatten(start_dim=1)[slicing]
        slice_error = loss_func(sliced_reconstruction, sliced_X)

        # compute its GGN diagonal ...
        with backpack(DiagGGNExact(), retain_graph=True):
            slice_error.backward(retain_graph=True)

        # ... and accumulate it
        for p_idx, p in enumerate(model.parameters()):
            diag_ggn_exact[p_idx] += p.diag_ggn_exact

    # fix normalization
    return [ggn / num_chunks for ggn in diag_ggn_exact]


# %%
#
# Let's benchmark peak memory and run time for different numbers of chunks:

num_chunks = [1, 4, 16, 64]

for n in num_chunks:
    print(f"GGN diagonal in {n} backward passes:")
    start = time()
    max_mem, diag_ggn_chunk = memory_usage(
        partial(diag_ggn_multiple_passes, n), interval=1e-3, max_usage=True, retval=True
    )
    end = time()
    print(f"\tPeak memory [MiB]: {max_mem:.2e}")
    print(f"\tTime [s]: {end-start:.2e}")

    correct = [
        allclose(diag1, diag2, rtol=5e-3, atol=5e-5)
        for diag1, diag2 in zip(diag_ggn, diag_ggn_chunk)
    ]
    print(f"\tCorrect: {correct}")

    if not all(correct):
        raise RuntimeError("Mismatch in GGN diagonals.")

# %%
#
# We can see that using more chunks consistently decreases the peak memory.
# Even run time decreases up to a sweet spot where increasing the number of
# chunks further eventually slows down the computation. The details of this
# trade-off will depend on your model and compute architecture.
#
# Concluding remarks
# ------------------
#
# Here, we considered chunking the computation along the auto-encoder's output
# dimension. There are other ways to achieve the desired effect of reducing
# peak memory:
#
# - In the mini-batch setting, we could only consider a subset of mini-batch
#   samples at each backpropagation. This can be done with the optional
#   :code:`subsampling` argument in many BackPACK's extensions. See the
#   :ref:`mini-batch sub-sampling tutorial <Mini-batch sub-sampling>`. This
#   technique can be combined with the above.
#
# - We could turn off the gradient computation (and thereby BackPACK's
#   computation) for all but a subgroup of parameters by setting their
#   :code:`requires_grad` attribute to :code:`False` and compute the GGN
#   diagonal only for these. However, for this to work we will need to perform
#   a new forward pass for each parameter subgroup.
