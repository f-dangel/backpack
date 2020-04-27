"""Test implementation of `sum_hessian`"""


import pytest
import torch
from torch.nn import CrossEntropyLoss, MSELoss

from backpack import extend

from .automated_test import check_sizes, check_values
from .test_sqrt_hessian import (
    autograd_hessian,
    classification_targets,
    derivative_from_layer,
    generate_data_input_hessian,
    make_id,
    regression_targets,
)

torch.manual_seed(0)
ARGS = "layer,input_shape,targets,raises"
SETTINGS = [
    # reduction mean
    [
        # layer
        CrossEntropyLoss(reduction="mean"),
        # input_shape
        (2, 3),
        # targets
        classification_targets(shape=(2,), num_classes=3),
        # tuple of exceptions that should be raised
        (),
    ],
    [
        CrossEntropyLoss(reduction="mean"),
        (8, 2),
        classification_targets(shape=(8,), num_classes=2),
        (),
    ],
    # reduction sum
    [
        CrossEntropyLoss(reduction="sum"),
        (2, 3),
        classification_targets(shape=(2,), num_classes=3),
        (),
    ],
    [
        CrossEntropyLoss(reduction="sum"),
        (8, 2),
        classification_targets(shape=(8,), num_classes=2),
        (),
    ],
    # non-scalar outputs are not supported
    [
        CrossEntropyLoss(reduction="none"),
        (8, 2),
        classification_targets(shape=(8,), num_classes=2),
        (ValueError,),
    ],
    # no reduction for a single number as input should be fine
    [
        CrossEntropyLoss(reduction="none"),
        (1, 1),
        classification_targets(shape=(1,), num_classes=1),
        (),
    ],
    # reduction mean
    [MSELoss(reduction="mean"), (5, 1), regression_targets(shape=(5, 1)), ()],
    [
        MSELoss(reduction="mean"),
        (5, 2),
        regression_targets(shape=(5, 2)),
        (NotImplementedError),
    ],
    # reduction sum
    [MSELoss(reduction="sum"), (5, 1), regression_targets(shape=(5, 1)), ()],
    [
        MSELoss(reduction="sum"),
        (5, 2),
        regression_targets(shape=(5, 2)),
        (NotImplementedError),
    ],
    # non-scalar outputs are not supported
    [
        MSELoss(reduction="none"),
        (5, 1),
        regression_targets(shape=(5, 1)),
        (ValueError,),
    ],
]
IDS = [make_id(layer, input_shape) for (layer, input_shape, _, _) in SETTINGS]


def autograd_sum_hessian(layer, input, targets):
    """Compute the Hessian of a loss module w.r.t. its input."""
    input.requires_grad = True
    loss = layer(input, targets)
    hessian = autograd_hessian(loss, input)

    sum_hessian = sum_hessian_blocks(hessian, input)

    return sum_hessian


def sum_hessian_blocks(hessian, input):
    """Sum second derivatives over the batch dimension.
    Assert second derivative w.r.t. different samples is zero.
    """
    num_axes = len(input.shape)

    if num_axes != 2:
        raise ValueError("Only 2D inputs are currently supported.")

    N = input.shape[0]
    num_features = input.numel() // N

    sum_hessian = torch.zeros(num_features, num_features)

    hessian_different_samples = torch.zeros(num_features, num_features)
    for n_1 in range(N):
        for n_2 in range(N):
            block = hessian[n_1, :, n_2, :]

            if n_1 == n_2:
                sum_hessian += block

            else:
                assert torch.allclose(block, hessian_different_samples)

    return sum_hessian


def backpack_sum_hessian(layer, input, targets):
    layer = extend(layer)
    derivative = derivative_from_layer(layer)

    # forward pass to initialize backpack buffers
    _ = layer(input, targets)

    sum_hessian = derivative.sum_hessian(layer, None, None)
    return sum_hessian


@pytest.mark.parametrize(ARGS, SETTINGS, ids=IDS)
def test_sum_hessian(layer, input_shape, targets, raises):
    """Compare the Hessian to reconstruction from individual Hessian sqrt."""
    torch.manual_seed(0)
    input = generate_data_input_hessian(input_shape)
    try:
        return _compare_sum_hessian(layer, input, targets)
    except Exception as e:
        if isinstance(e, raises):
            return
        else:
            raise e


def _compare_sum_hessian(layer, input, targets):
    backpack_result = backpack_sum_hessian(layer, input, targets)
    autograd_result = autograd_sum_hessian(layer, input, targets)

    check_sizes(autograd_result, backpack_result)
    check_values(autograd_result, backpack_result)

    return backpack_result
