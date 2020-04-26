""" Test implementation of `sqrt_hessian_sampled` for loss derivatives. """


import pytest
import torch
from torch.nn import CrossEntropyLoss, MSELoss

from backpack import extend

from .automated_test import check_sizes, check_values
from .test_sqrt_hessian import (
    autograd_input_hessian,
    classification_targets,
    derivative_from_layer,
    embed,
    generate_data_input_hessian,
    make_id,
    regression_targets,
    sample_hessians_from_sqrt,
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
    # reduction sum
    [MSELoss(reduction="sum"), (5, 1), regression_targets(shape=(5, 1)), ()],
    # no sampling implemented yet
    [
        MSELoss(reduction="sum"),
        (5, 2),
        regression_targets(shape=(5, 2)),
        (NotImplementedError,),
    ],
    [
        MSELoss(reduction="mean"),
        (5, 2),
        regression_targets(shape=(5, 2)),
        (NotImplementedError,),
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


def backpack_hessian_via_sqrt_hessian(layer, input, targets):
    layer = extend(layer)
    derivative = derivative_from_layer(layer)

    # forward pass to initialize backpack buffers
    _ = layer(input, targets)

    sqrt_hessian = derivative.sqrt_hessian(layer, None, None)
    individual_hessians = sample_hessians_from_sqrt(sqrt_hessian)

    hessian = embed(individual_hessians, input)

    return hessian


def backpack_hessian_via_sqrt_hessian_sampled(layer, input, targets):
    MC_SAMPLES = 100000

    layer = extend(layer)
    derivative = derivative_from_layer(layer)

    # forward pass to initialize backpack buffers
    _ = layer(input, targets)

    sqrt_hessian = derivative.sqrt_hessian_sampled(
        layer, None, None, mc_samples=MC_SAMPLES
    )
    individual_hessians = sample_hessians_from_sqrt(sqrt_hessian)

    # embed
    # TODO improve readability
    hessian_shape = (*input.shape, *input.shape)
    hessian = torch.zeros(hessian_shape)

    hessian = embed(individual_hessians, input)

    return hessian


@pytest.mark.parametrize(ARGS, SETTINGS, ids=IDS)
def test_sqrt_hessian_via_input_hessian_sampled(layer, input_shape, targets, raises):
    """Compare the Hessian to reconstruction from individual sampled Hessian sqrt."""
    torch.manual_seed(0)
    input = generate_data_input_hessian(input_shape)
    try:
        return _compare_input_hessian_sampled(layer, input, targets)
    except Exception as e:
        if isinstance(e, raises):
            return
        else:
            raise e


def _compare_input_hessian_sampled(layer, input, targets):
    RTOL = 1e-2
    backpack_result = backpack_hessian_via_sqrt_hessian_sampled(layer, input, targets)
    autograd_result = autograd_input_hessian(layer, input, targets)

    check_sizes(autograd_result, backpack_result)
    check_values(autograd_result, backpack_result, rtol=RTOL)

    return backpack_result
