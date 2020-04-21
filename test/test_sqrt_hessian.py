""" Test implementation of `sqrt_hessian` for loss derivatives. """


import pytest
import torch
from torch.nn import CrossEntropyLoss  # , MSELoss

from backpack import extend
from backpack.core.derivatives import derivatives_for
from backpack.hessianfree.hvp import hessian_vector_product

from .automated_test import check_sizes, check_values


def classification_targets(shape, num_classes):
    """Create random targets for classes 0, ..., `num_classes - 1`."""
    return torch.randint(size=shape, low=0, high=num_classes)


def regression_targets(shape):
    """Create random targets for regression."""
    return torch.rand(size=shape)


def make_id(layer, input_shape):
    """Create an id for the test scenario."""
    return "in{}-{}".format(input_shape, layer)


torch.manual_seed(0)
ARGS = "layer,input_shape,targets"
SETTINGS = [
    # (layer, input_shape, targets)
    [
        CrossEntropyLoss(reduction="mean"),
        (2, 3),
        classification_targets(shape=(2,), num_classes=3),
    ],
    [
        CrossEntropyLoss(reduction="mean"),
        (8, 2),
        classification_targets(shape=(8,), num_classes=2),
    ],
    # [MSELoss(reduction="mean"), (1, 5), None],
    # [MSELoss(reduction="sum"), (2, 6), None],
]
IDS = [make_id(layer, input_shape) for (layer, input_shape, _) in SETTINGS]


def autograd_hessian(loss, x):
    """Return the Hessian matrix of `loss` w.r.t. `x`.

    Arguments:
        loss (torch.Tensor): A scalar-valued tensor.
        x (torch.Tensor): Tensor used in the computation graph of `loss`.

    Shapes:
        loss: `[1,]`
        x: `[A, B, C, ...]`

    Returns:
        torch.Tensor: Hessian tensor of `loss` w.r.t. `x`. The Hessian has shape
            `[A, B, C, ..., A, B, C, ...]`.
    """
    assert loss.shape == torch.Size([])
    vectorized_shape = (x.numel(), x.numel())
    final_shape = (*x.shape, *x.shape)

    hessian_vec_x = torch.zeros(vectorized_shape)

    num_cols = hessian_vec_x.shape[1]
    for column_idx in range(num_cols):
        unit = torch.zeros(num_cols)
        unit[column_idx] = 1.0

        unit = unit.view_as(x)
        column = hessian_vector_product(loss, [x], [unit])[0].reshape(-1)

        hessian_vec_x[:, column_idx] = column

    return hessian_vec_x.reshape(final_shape)


def autograd_input_hessian(layer, input, targets):
    """Compute the Hessian of a loss module w.r.t. its input."""
    input.requires_grad = True

    loss = layer(input, targets)
    return autograd_hessian(loss, input)


# TODO remove duplicate
def derivative_from_layer(layer):
    layer_to_derivative = derivatives_for

    for module_cls, derivative_cls in layer_to_derivative.items():
        if isinstance(layer, module_cls):
            return derivative_cls()

    raise RuntimeError("No derivative available for {}".format(layer))


def backpack_input_hessian_with_sqrt_hessian(layer, input, targets):
    layer = extend(layer)
    derivative = derivative_from_layer(layer)

    # forward pass to initialize backpack buffers
    _ = layer(input, targets)

    sqrt_hessian = derivative.sqrt_hessian(layer, None, None)
    individual_hessians = sample_hessians_from_sqrt(sqrt_hessian)

    # embed
    # TODO improve readability
    hessian_shape = (*input.shape, *input.shape)
    hessian = torch.zeros(hessian_shape)

    N = input.shape[0]

    for n in range(N):
        num_axes = len(input.shape)

        if num_axes == 2:
            hessian[n, :, n, :] = individual_hessians[n]
        else:
            raise ValueError("Only 2D inputs are currently supported.")

    return hessian


def sample_hessians_from_sqrt(sqrt):
    """Convert individual matrix square root into individual full matrix. """
    equation = None
    num_axes = len(sqrt.shape)

    # TODO improve readability
    if num_axes == 3:
        equation = "vni,vnj->nij"
    else:
        raise ValueError("Only 2D inputs are currently supported.")

    return torch.einsum(equation, sqrt, sqrt)


def generate_data_input_hessian(input_shape):
    input = torch.rand(input_shape)
    return input


@pytest.mark.parametrize(ARGS, SETTINGS, ids=IDS)
def test_input_hessian(layer, input_shape, targets):
    torch.manual_seed(0)
    input = generate_data_input_hessian(input_shape)
    return _compare_input_hessian(layer, input, targets)


def _compare_input_hessian(layer, input, targets):
    autograd_result = autograd_input_hessian(layer, input, targets)
    backpack_result = backpack_input_hessian_with_sqrt_hessian(layer, input, targets)

    check_sizes(autograd_result, backpack_result)
    check_values(autograd_result, backpack_result)

    return backpack_result
