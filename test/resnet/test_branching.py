"""This test demonstrates the custom modules in backpack for branching/ResNets.

For torch < 1.9.0 (no full backward hook), this test is important,
since those fail without ActiveIdentity.
For torch>=1.9.0 (full backward hook), all tests pass.
Additionally, for torch>=1.9.0 there is a convenient option use_converter=True in extend().
As soon as torch version is bumped to 1.9.0 this test can be deleted.
"""
from contextlib import nullcontext
from test.automated_test import check_sizes_and_values
from test.core.derivatives.utils import classification_targets
from typing import Callable, Tuple

from pytest import mark, raises
from torch import Tensor, cat, manual_seed, rand, zeros
from torch.nn import (
    CrossEntropyLoss,
    Identity,
    Linear,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
)
from torch.nn.utils.convert_parameters import parameters_to_vector

from backpack import backpack, extend, extensions
from backpack.custom_module.branching import ActiveIdentity, Parallel
from backpack.hessianfree.ggnvp import ggn_vector_product
from backpack.utils import FULL_BACKWARD_HOOK, exception_inside_backward_pass
from backpack.utils.convert_parameters import vector_to_parameter_list


def setup(
    apply_extend: bool = False, active_identity: bool = True
) -> Tuple[Tensor, Tensor, Module, Module]:
    """Set seed. Generate and return inputs, labels, model and loss function.

    A simple ResNet using the ``Branch`` and ``SumModule`` modules to handle branching.

    Args:
        active_identity: Whether the identity function should create a new node
            in the computation graph.
        apply_extend: Whether model and loss function should be extended.

    Returns:
        X, y, model, loss_function
    """
    manual_seed(0)

    N = 7

    in_features = 20
    hidden_features = 10
    out_features = 3

    X = rand((N, in_features))
    y = classification_targets((N,), out_features)

    identity = ActiveIdentity() if active_identity else Identity()

    model = Sequential(
        Linear(in_features, hidden_features),
        ReLU(),
        # skip connection
        Parallel(
            identity,
            Linear(hidden_features, hidden_features),
        ),
        # end of skip connection
        Sigmoid(),
        Linear(hidden_features, out_features),
    )
    loss_function = CrossEntropyLoss(reduction="mean")

    if apply_extend:
        model = extend(model, debug=True)
        loss_function = extend(loss_function, debug=True)

    return X, y, model, loss_function


def setup_convenient(
    apply_extend: bool = False, active_identity: bool = True
) -> Tuple[Tensor, Tensor, Module, Module]:
    """Set seed. Generate and return inputs, labels, model and loss function.

    A simple ResNet using the ``Parallel`` convenience module around the ``Branch`` and
    ``SumModule`` modules to handle branching.

    Args:
        active_identity: Whether the identity function should create a new node
            in the computation graph.
        apply_extend: Whether model and loss function should be extended.

    Returns:
        X, y, model, loss_function
    """
    manual_seed(0)

    N = 7

    in_features = 20
    hidden_features = 10
    out_features = 3

    X = rand((N, in_features))
    y = classification_targets((N,), out_features)

    identity = ActiveIdentity() if active_identity else Identity()

    model = Sequential(
        Linear(in_features, hidden_features),
        ReLU(),
        # skip connection
        Parallel(
            identity,
            Linear(hidden_features, hidden_features),
        ),
        # end of skip connection
        Sigmoid(),
        Linear(hidden_features, out_features),
    )
    loss_function = CrossEntropyLoss(reduction="mean")

    if apply_extend:
        model = extend(model, debug=True)
        loss_function = extend(loss_function, debug=True)

    return X, y, model, loss_function


def autograd_diag_ggn_exact(
    X: Tensor, y: Tensor, model: Module, loss_function: Module
) -> Tensor:
    """Compute the generalized Gauss-Newton diagonal via autodiff.

    Args:
        X: data
        y: target
        model: model
        loss_function: loss function

    Returns:
        diag_ggn
    """
    D = sum(p.numel() for p in model.parameters())

    outputs = model(X)
    loss = loss_function(outputs, y)

    ggn_diag = zeros(D)

    # compute GGN columns by GGNVPs with one-hot vectors
    for d in range(D):
        e_d = zeros(D)
        e_d[d] = 1.0
        e_d_list = vector_to_parameter_list(e_d, model.parameters())

        ggn_d_list = ggn_vector_product(loss, outputs, model, e_d_list)

        ggn_diag[d] = parameters_to_vector(ggn_d_list)[d]

    return ggn_diag


def backpack_diag_ggn_exact(
    X: Tensor, y: Tensor, model: Module, loss_function: Module
) -> Tensor:
    """Compute the generalized Gauss-Newton diagonal via BackPACK.

    Args:
        X: data
        y: target
        model: model
        loss_function: loss function

    Returns:
        diag_ggn_exact
    """
    outputs = model(X)
    loss = loss_function(outputs, y)

    with backpack(extensions.DiagGGNExact(), debug=True):
        loss.backward()

    return cat([p.diag_ggn_exact.flatten() for p in model.parameters()])


SETUPS = [setup, setup_convenient]
SETUPS_IDS = ["simple-resnet", "simple-resnet-convenient"]


@mark.parametrize("setup_fn", SETUPS, ids=SETUPS_IDS)
def test_diag_ggn_exact_active_identity(setup_fn: Callable) -> None:
    """Compare diagonal GGN of a ResNet.

    Args:
        setup_fn: setup function
    """
    X, y, model, loss_function = setup_fn()

    autograd_result = autograd_diag_ggn_exact(X, y, model, loss_function)

    X, y, model, loss_function = setup_fn(apply_extend=True)

    backpack_result = backpack_diag_ggn_exact(X, y, model, loss_function)

    check_sizes_and_values(autograd_result, backpack_result)


@mark.parametrize("setup_fn", SETUPS, ids=SETUPS_IDS)
def test_diag_ggn_exact_nn_identity_fails(setup_fn: Callable) -> None:
    """``torch.nn.Identity`` does not create a node and messes up backward hooks.

    However, it works fine if using full backward hook. (torch >= 1.9.0)

    Args:
        setup_fn: setup function
    """
    X, y, model, loss_function = setup_fn(active_identity=False)

    autograd_result = autograd_diag_ggn_exact(X, y, model, loss_function)

    X, y, model, loss_function = setup_fn(apply_extend=True, active_identity=False)

    with nullcontext() if FULL_BACKWARD_HOOK else raises(
        exception_inside_backward_pass(AssertionError)
    ):
        backpack_result = backpack_diag_ggn_exact(X, y, model, loss_function)

        check_sizes_and_values(autograd_result, backpack_result)
