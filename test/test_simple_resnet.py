"""An example to check if BackPACK' first-order extensions are working for ResNets."""

from test.core.derivatives.utils import classification_targets

import torch

from backpack import backpack, extend, extensions

from .automated_test import check_sizes, check_values


def autograd_individual_gradients(X, y, model, loss_func):
    """Individual gradients via for loop with automatic differentiation.

    Args:
        X (torch.Tensor): Mini-batch of shape `(N, *)`
        y (torch.Tensor: Labels for `X`
        model (torch.nn.Module): Model for forward pass
        loss_func (torch.nn.Module): Loss function for model prediction

    Returns:
        [torch.Tensor]: Individual gradients for samples in the mini-batch
            with respect to the model parameters. Arranged in the same order
           as `model.parameters()`.
    """
    N = X.shape[0]
    reduction_factor = _get_reduction_factor(X, y, model, loss_func)

    individual_gradients = [
        torch.zeros(N, *p.shape).to(X.device) for p in model.parameters()
    ]

    for n in range(N):
        x_n = X[n].unsqueeze(0)
        y_n = y[n].unsqueeze(0)

        f_n = model(x_n)
        l_n = loss_func(f_n, y_n) / reduction_factor

        g_n = torch.autograd.grad(l_n, model.parameters())

        for idx, g in enumerate(g_n):
            individual_gradients[idx][n] = g

    return individual_gradients


def _get_reduction_factor(X, y, model, loss_func):
    """Return reduction factor of loss function."""
    N = X.shape[0]

    x_0 = X[0].unsqueeze(0)
    y_0 = y[0].unsqueeze(0)

    x_0_repeated = x_0.repeat([N if pos == 0 else 1 for pos, _ in enumerate(X.shape)])
    y_0_repeated = y_0.repeat([N if pos == 0 else 1 for pos, _ in enumerate(y.shape)])

    individual_loss = loss_func(model(x_0), y_0)
    reduced_loss = loss_func(model(x_0_repeated), y_0_repeated)

    return (N * individual_loss / reduced_loss).item()


def backpack_individual_gradients(X, y, model, loss_func):
    """Individual gradients with BackPACK.

    Args:
        X (torch.Tensor): Mini-batch of shape `(N, *)`
        y (torch.Tensor: Labels for `X`
        model (torch.nn.Module): Model for forward pass
        loss_func (torch.nn.Module): Loss function for model prediction

    Returns:
        [torch.Tensor]: Individual gradients for samples in the mini-batch
            with respect to the model parameters. Arranged in the same order
           as `model.parameters()`.
    """
    model = extend(model)
    loss_func = extend(loss_func)

    loss = loss_func(model(X), y)

    with backpack(extensions.BatchGrad()):
        loss.backward()

    individual_gradients = [p.grad_batch for p in model.parameters()]

    return individual_gradients


class Identity(torch.nn.Module):
    """Identity operation."""

    def forward(self, input):
        return input


class Parallel(torch.nn.Sequential):
    """Feed input to multiple modules, sum the result.

              |-----|
        | ->  | f_1 |  -> |
        |     |-----|     |
        |                 |
        |     |-----|     |
    x ->| ->  | f_2 |  -> + -> f₁(x) + f₂(x) + ...
        |     |-----|     |
        |                 |
        |     |-----|     |
        | ->  | ... |  -> |
              |-----|

    """

    def forward(self, input):
        """Process input with all modules, sum the output."""
        for idx, module in enumerate(self.children()):
            if idx == 0:
                output = module(input)
            else:
                output = output + module(input)

        return output


def test_individual_gradients_simple_resnet():
    """Individual gradients for a simple ResNet with autodiff and BackPACK."""

    # batch size, feature dimension
    N, D = 2, 5
    # classification
    C = 3

    X = torch.rand(N, D)
    y = classification_targets((N,), num_classes=C)

    model = Parallel(Identity(), torch.nn.Linear(D, D, bias=True))
    loss_func = torch.nn.CrossEntropyLoss(reduction="sum")

    result_autograd = autograd_individual_gradients(X, y, model, loss_func)
    result_backpack = backpack_individual_gradients(X, y, model, loss_func)

    check_sizes(result_autograd, result_backpack)
    check_values(result_autograd, result_backpack)
