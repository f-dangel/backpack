r"""Gradient of backpropagated quantities
=========================================

If :py:meth:`backward() <torch.Tensor.backward>` is called with
``create_graph=True``, PyTorch creates the computation graph of the outputs
of the backward pass, including quantities computed by BackPACK.
This makes it possible to compute higher order derivatives with PyTorch,
even if BackPACK's extensions no longer apply.

.. warning::

    This feature should work with any BackPACK extension, but has not been
    extensively tested and should be considered experimental. We recommend
    that you test your specific setup before running large scale experiments.
    Please `get in touch <https://github.com/f-dangel/backpack/issues>`_
    if something does not look right.

This example show how to compute
the gradient of (the total variance of (the individual gradients)),
along with some sanity checks.

"""

# %%
# Let's get the imports and configuration out of the way.

import torch

from backpack import backpack, extend
from backpack.extensions import Variance
from backpack.utils.examples import load_one_batch_mnist
from torch import nn

torch.manual_seed(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# %%
# We'll start with some functions to compute the total variance,
# the sum of the variance along each dimension of the individual gradients,
# and it's gradient.
#
# We'll assume that the data will be given as
#
# .. code::
#
#     # x     : a [N, ...] tensor of inputs
#     # y     : a [N, ...] tensor of targets
#     # model : an model extended with BackPACK that takes x as input
#     # loss  : a loss function that takes model(x) and y as input
#
# such that the loss is given by ``loss(model(x), y)``.


def total_variance_and_gradient_backpack(x, y, model, lossfunc):
    """Computes the total variance of the individual gradients and its gradient.

    Uses BackPACK's :py:meth:`Variance <backpack.extensions.Variance>`
    and PyTorch's :py:meth:`backward() <torch.Tensor.backward>`
    pass with the argument ``create_graph=True``.
    """
    model.zero_grad()
    loss = lossfunc(model(x), y)
    with backpack(Variance()):
        loss.backward(retain_graph=True, create_graph=True)

    total_var = 0
    for p in model.parameters():
        total_var += torch.sum(p.variance)

    grad_of_var = torch.autograd.grad(total_var, model.parameters())

    return total_var, grad_of_var


# %%
#
#


def individual_gradients_pytorch(x, y, model, lossfunc):
    """Computes the tensor of individual gradients using PyTorch.

    Iterates over the samples to compute individual gradients.
    Flattens and concatenates the individual gradients to return
    a ``[N, D]`` tensor where

    - ``N`` is the number of samples
    - ``D`` is the total number of parameters

    Calls :py:meth:`backward <torch.autograd.backward>` with ``create_graph=True``
    to make it possible to backpropagate through the gradients again.
    """
    model.zero_grad()
    grads_vector_format = []
    for i in range(x.shape[0]):
        x_i = x[i, :].unsqueeze(0)
        if len(y.shape) == 1:
            y_i = y[i].unsqueeze(0)
        else:
            y_i = y[i, :].unsqueeze(0)

        loss = lossfunc(model(x_i), y_i)

        grad_list_format = torch.autograd.grad(
            loss, model.parameters(), create_graph=True, retain_graph=True
        )
        grad_vector_format = torch.cat([g.view(-1,) for g in grad_list_format])
        grads_vector_format.append(grad_vector_format.clone())

    return torch.stack(grads_vector_format)


def total_variance_and_gradient_pytorch(x, y, model, lossfunc):
    """Computes the total variance of the individual gradients and its gradient.

    Uses ``individual_gradients_pytorch`` to compute the individual gradients.
    """
    ind_grads = individual_gradients_pytorch(x, y, model, lossfunc)
    variance = torch.var(ind_grads, dim=0, unbiased=False)
    total_var = torch.sum(variance)

    grad_of_var = torch.autograd.grad(total_var, model.parameters())

    return total_var, grad_of_var


# %%
# Let's write a quick test to check whether the results returned by BackPACK
# and PyTorch match, up to some precision. It's not possible to get the same
# result up to high precision without using higher floating point precision
# (:py:obj:`torch.Tensor.double`) as the two procedures do sums in different
# orders and have different rounding errors.


def check_same_results(x, y, model, lossfunc):
    """Compares the results between the pytorch and backpack implementations."""
    var_bp, grad_var_bp = total_variance_and_gradient_backpack(x, y, model, lossfunc)
    var_pt, grad_var_pt = total_variance_and_gradient_pytorch(x, y, model, lossfunc)

    print("Total variance is the same?")
    print("   ", torch.allclose(var_bp, var_pt, atol=1e-4, rtol=1e-4))

    print("Variance of the total variance w.r.t. parameters is the same?")
    for (name, _), p_grad_var_bp, p_grad_var_pt in zip(
        model.named_parameters(), grad_var_bp, grad_var_pt
    ):
        match = torch.allclose(p_grad_var_bp, p_grad_var_pt, atol=1e-4, rtol=1e-4)
        print("    {:<20}: {}".format(name, match))


# %%
# We can now test some models. Let's start with something simple,
# a linear model with 3 parameters on artificial data.
#

N, D = 3, 2
x = torch.randn(N, D).to(DEVICE)
y = torch.randn(N, 1).to(DEVICE)
model = extend(nn.Sequential(nn.Linear(D, 1, bias=False))).to(DEVICE)
lossfunc = torch.nn.MSELoss(reduction="sum")

check_same_results(x, y, model, lossfunc)

# %%
# We can also try a linear model on MNIST data
#


x, y = load_one_batch_mnist(batch_size=32)
x, y = x.to(DEVICE), y.to(DEVICE)

model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10))
model = extend(model).to(DEVICE)

lossfunc = torch.nn.CrossEntropyLoss(reduction="sum")

check_same_results(x, y, model, lossfunc)

# %%
# And a small CNN for some architecture variety
#


x, y = load_one_batch_mnist(batch_size=32)
x, y = x.to(DEVICE), y.to(DEVICE)

model = extend(
    torch.nn.Sequential(
        torch.nn.Conv2d(1, 5, 5, 1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Flatten(),
        torch.nn.Linear(720, 10),
    )
).to(DEVICE)

lossfunc = torch.nn.CrossEntropyLoss(reduction="sum")

check_same_results(x, y, model, lossfunc)
