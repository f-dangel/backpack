"""
Differential privacy example
=============================

Compute the gradient with PyTorch and other quantities with BackPACK.
"""

import torch
from torch.optim import Optimizer
import torch.nn as nn
from backpack import backpack, extend
from backpack.extensions import BatchGrad, BatchL2Grad
from backpack.utils.examples import get_mnist_dataloder
import matplotlib.pyplot as plt

# %%
# Script configuration
# -----------------------------

NUM_EPOCHS = 1
BATCH_SIZE = 512
PRINT_EVERY = 50
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


# %%
# Some utility functions
# -------------------------


def make_broadcastable(v, X):
    """Returns a view of `v` that can be broadcast with `X`

    If `v` is a one-dimensional tensor [N] and `X` is a tensor of shape
    `[N, ..., ]`, returns a view of v with singleton dimension appended,
    such that `v` and `X` have the same number of dimensions
    """
    broadcasting_shape = (-1, *[1 for _ in X.shape[1:]])
    return v.reshape(broadcasting_shape)


def get_accuracy(output, targets):
    """Helper function to print the accuracy"""
    predictions = output.argmax(dim=1, keepdim=True).view_as(targets)
    return predictions.eq(targets).float().mean().item()


# %%
# Creating the model and loading data
# -----------------------------------


def make_small_cnn(outputs=10, channels=(16, 32), fc_dim=32, kernels=(8, 4)):
    return nn.Sequential(
        nn.ZeroPad2d((3, 4, 3, 4)),
        nn.Conv2d(1, channels[0], kernels[0], stride=2, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=1),
        nn.Conv2d(channels[0], channels[1], kernels[1], stride=2, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=1),
        nn.Flatten(),
        nn.Linear(channels[1] * 4 * 4, fc_dim),
        nn.ReLU(),
        nn.Linear(fc_dim, outputs),
    )


mnist_dataloader = get_mnist_dataloder()

model = make_small_cnn().to(DEVICE)
loss_function = nn.CrossEntropyLoss().to(DEVICE)

model = extend(model)
loss_function = extend(loss_function)

# %%
# How to access the data needed to compute the update
# ---------------------------------------------------
#
# Let's take a single batch
#
# The update needs the following operations
#     gᵢ = ∇f(xᵢ, yᵢ, w) for each i in the minibatch
#
#     ̃gᵢ = gᵢ × max(1, ‖gᵢ‖₂/C)
#
#     ̃g = ∑ᵢ ̃gᵢ


x, y = next(iter(mnist_dataloader))
x = x.to(DEVICE)
y = y.to(DEVICE)

loss = loss_function(model(x), y)
with backpack(BatchL2Grad(), BatchGrad()):
    loss.backward()

maximum_grad_norm = 1
C = maximum_grad_norm

for p in model.parameters():
    print(p.grad.shape, p.batch_l2.shape, p.grad_batch.shape)

l2_norms_squared_all_params_list = [p.batch_l2 for p in model.parameters()]
l2_norms_squared_all_params = torch.stack(l2_norms_squared_all_params_list)
l2_norms = torch.sqrt(torch.sum(l2_norms_squared_all_params, dim=0))
scaling_factors = torch.clamp_max(l2_norms / C, 1.0)

for p in model.parameters():
    broadcasting_shape = (-1, *[1 for dim in p.grad_batch.shape[1:]])
    clipped_grads = p.grad_batch * scaling_factors.reshape(broadcasting_shape)
    clipped_grad = torch.sum(clipped_grads, dim=0)


# %%
# Writing the optimizer
# ---------------------


class DP_SGD(Optimizer):
    """
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)

    """

    def __init__(self, params, lr=0.1, max_norm=0.01, stddev=1.1):
        self.lr = lr
        self.max_norm = max_norm
        self.stddev = stddev
        super().__init__(params, dict())

    def step(self):
        """Performs a single optimization step."""

        l2_norms_all_params_list = []
        for group in self.param_groups:
            for p in group["params"]:
                l2_norms_all_params_list.append(p.batch_l2)

        l2_norms_all_params = torch.stack(l2_norms_all_params_list)
        total_norms = torch.sqrt(torch.sum(l2_norms_all_params, dim=0))
        scaling_factors = torch.clamp_max(total_norms / self.max_norm, 1.0)

        for group in self.param_groups:
            for p in group["params"]:
                clipped_grads = p.grad_batch * make_broadcastable(
                    scaling_factors, p.grad_batch
                )
                clipped_grad = torch.sum(clipped_grads, dim=0)

                noise_magnitude = 2 * self.stddev * self.max_norm
                noise = torch.randn_like(clipped_grad) * noise_magnitude

                perturbed_update = clipped_grad + noise

                p.data.add_(-self.lr * perturbed_update)


# %%
# Do the optimization
# --------------------

optimizer = DP_SGD(model.parameters(), lr=0.1, max_norm=0.01, stddev=1.1)

losses = []
accuracies = []
for epoch in range(NUM_EPOCHS):
    for batch_idx, (x, y) in enumerate(mnist_dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        outputs = model(x)
        loss = loss_function(outputs, y)
        model.zero_grad()

        with backpack(BatchGrad(), BatchL2Grad()):
            loss.backward()

        optimizer.step()

        should_print = (batch_idx % PRINT_EVERY) == 0
        losses.append(loss.detach().item())
        accuracies.append(get_accuracy(outputs, y))
        if should_print:
            print(
                "Epoch %3.d/%d Iteration %3.d " % (epoch, NUM_EPOCHS, batch_idx)
                + "Minibatch Loss %.3f  " % losses[-1]
                + "Accuracy %.3f" % accuracies[-1]
            )
        if batch_idx > 200:
            break

fig = plt.figure()
axes = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]

axes[0].plot(losses)
axes[0].set_title("Loss")
axes[0].set_xlabel("Iteration")

axes[1].plot(accuracies)
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Iteration")
