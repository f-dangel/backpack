"""Save memory in convolutions
==============================

There exist different approaches to apply the Jacobian with respect to the kernel
of a convolution. They exhibit a non-trivial trade-off between run time and memory
consumption (see more details below). The default choice in BackPACK is a memory-
intensive implementation. This can lead to an out-of-memory error.

Here, we show how to switch BackPACK's vector-Jacobian product algorithm for the kernel
(``weight``) of :py:class:`torch.nn.Conv2d` modules to a memory-saving variant
presented in `[Rochette, 2019] <https://arxiv.org/abs/1912.06015>`_.

This can be helpful if you are experiencing memory overflows in CNNs.

.. note ::
    This feature is experimental and may change in future releases.

.. note ::
    Currently, the savings only affect BackPACK's first-order extensions.
    This may change in future releases.

"""

# %%
# Let's get the imports out of our way.

import time

import torch
from memory_profiler import memory_usage
from torch.nn import (
    Conv2d,
    CrossEntropyLoss,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)

from backpack import backpack, extend, extensions
from backpack.core.derivatives.convnd import weight_jac_t_save_memory
from backpack.utils.examples import load_one_batch_mnist

# %%
# We start with some utilities for setting up an extended CNN, loss function, and
# input data from the MNIST data set.


def setup(device):
    """Load MNIST batch, create extended CNN and loss function. Load to device.

    Args:
        device (torch.device): Device that all objects are transferred to.

    Returns:
        inputs, labels, model, loss function
    """
    X, y = load_one_batch_mnist(batch_size=128)
    X, y = X.to(device), y.to(device)

    model = extend(
        Sequential(
            Conv2d(1, 128, 3, padding=1),
            ReLU(),
            MaxPool2d(3, stride=2),
            Conv2d(128, 256, 3, padding=1),
            ReLU(),
            MaxPool2d(3, padding=1, stride=2),
            Conv2d(256, 64, 3, padding=1),
            ReLU(),
            MaxPool2d(3, stride=2),
            Conv2d(64, 32, 3, padding=1),
            ReLU(),
            MaxPool2d(3, stride=2),
            Flatten(),
            Linear(32, 10),
        ).to(device)
    )

    lossfunc = extend(CrossEntropyLoss().to(device))

    return X, y, model, lossfunc


# %%
# Let's demonstrate the differences between the vector-Jacobian methods, we benchmark
# a function that computes individual gradients on the specified setup using BackPACK's
# :py:class:`BatchGrad <backpack.extensions.BatchGrad>` extensions.


def compute_individual_gradients(device, seed=0):
    """Compute individual gradients for the seeded problem specified in ``setup``.

    Args:
        device (torch.device): Device that the computation should be performed on.
        seed (int): Random seed to set before setting up the problem.

    Return a dictionary with parameter name and individual gradients as key value
    pairs.
    """
    torch.manual_seed(seed)

    X, y, model, lossfunc = setup(device)

    loss = lossfunc(model(X), y)

    with backpack(extensions.BatchGrad()):
        loss.backward()

    return {name: param.grad_batch for name, param in model.named_parameters()}


# %%
# The memory-saving strategy for application of the transposed Jacobian can be chosen
# by wrapping the backward pass with BackPACK into a
# choice :py:class:`weight_jac_t_save_memory<backpack.core.derivatives.convnd.weight_jac_t_save_memory>`
# context, which accepts the self-explanatory argument ``save_memory``.

# %%
# Peak memory comparison
# ----------------------
# Let's see the differences between both vector-Jacobian methods in terms of peak
# memory consumption.


def compare_peakmem(device):
    """Print peak memory of different vector-Jacobian algorithms for convolution.

    Peak memory only makes sense when ``device`` is CPU as memory usage on GPU
    cannot be tracked by this implementation.

    Args:
        device (torch.device): Device that the computation should be performed on.
    """
    print(f"Device: {device}")

    for save_memory in True, False:

        with weight_jac_t_save_memory(save_memory=save_memory):
            func = lambda: compute_individual_gradients(device)
            interval = 1e-3
            peakmem = max(memory_usage(func, interval=interval))

        print(f"Save memory: {save_memory}\tPeak memory: {peakmem:.1f}")


compare_peakmem(torch.device("cpu"))

# %%
# As expected, the backpropagation with ``save_memory=True`` requires less memory.


# %%
# Run time comparison
# -------------------
# Next, we inspect the run time of both save-memory strategies.


def compare_runtime(device):
    """Print run time of different vector-Jacobian algorithms for convolution.

    Args:
        device (torch.device): Device that the computation should be performed on.
    """
    print(f"Device: {device}")

    for save_memory in True, False:

        with weight_jac_t_save_memory(save_memory=save_memory):
            start = time.time()

            compute_individual_gradients(device)
            torch.cuda.synchronize()

            run_time = time.time() - start

        print(f"Save memory: {save_memory}\tRun time: {run_time:.3f}")


compare_runtime(torch.device("cpu"))

# %%
# In this case, saving memory comes at the cost of reduced run time performance.

# %%
# If you have a GPU, you can also see the differences there, too:

if torch.cuda.is_available():
    compare_runtime(torch.device("cuda"))

# %%
# Let's quickly double-check that both algorithms computed the same result.
device = torch.device("cpu")

with weight_jac_t_save_memory(save_memory=True):
    individual_gradients = compute_individual_gradients(device)

with weight_jac_t_save_memory(save_memory=False):
    save_memory_individual_gradients = compute_individual_gradients(device)

print(f"{'Parameter':<10}| Same individual gradients?")
for param_name in individual_gradients.keys():

    same = torch.allclose(
        individual_gradients[param_name],
        save_memory_individual_gradients[param_name],
        atol=1e-7,
    )
    msg = f"{param_name:<10}| {same}"

    if same:
        print(msg)
    else:
        raise ValueError(msg)

# %%
# Some heuristics
# ---------------
# So, when should you make use of ``save_memory``?
# TODO
