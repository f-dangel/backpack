"""Checks wether backpack correctly recognizes when 2nd order backprop would fail.

Failure cases include
- the loss if not the output of a torch.nn module
- using unsupported parameters of the loss
"""

import pytest
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from backpack import extend
from backpack import backpack as bp
import backpack.extensions as bpext

ext_2nd_order = [
    bpext.KFAC,
    bpext.KFRA,
    bpext.KFLR,
    bpext.DiagGGNExact,
    bpext.DiagGGNMC,
]
ext_2nd_order_name = [
    "KFAC",
    "KFRA",
    "KFLR",
    "DiagGGNExact",
    "DiagGGNExactMC",
]


def classification_targets(N, num_classes):
    """Create random targets for classes 0, ..., `num_classes - 1`."""
    return torch.randint(size=(N,), low=0, high=num_classes)


def dummy_cross_entropy(N=5):
    y_pred = torch.rand((N, 2))
    y_pred.requires_grad = True
    y = classification_targets(N, 2)
    loss_module = extend(CrossEntropyLoss())
    return loss_module(y_pred, y)


def dummy_mse(N=5, D=1):
    y_pred = torch.rand((N, D))
    y_pred.requires_grad = True
    y = torch.randn((N, D))
    loss_module = extend(MSELoss())
    return loss_module(y_pred, y)


@pytest.mark.parametrize("extension", ext_2nd_order, ids=ext_2nd_order_name)
def test_sqrt_hessian_crossentropy_should_pass(extension):
    loss = dummy_cross_entropy()

    with bp(extension()):
        loss.backward()


@pytest.mark.parametrize("extension", ext_2nd_order, ids=ext_2nd_order_name)
def test_sqrt_hessian_mse_should_pass(extension):
    loss = dummy_mse()

    with bp(extension()):
        loss.backward()


@pytest.mark.parametrize("extension", ext_2nd_order, ids=ext_2nd_order_name)
def test_sqrt_hessian_modified_crossentropy_should_fail(extension):
    loss = dummy_cross_entropy() * 2

    with pytest.warns(UserWarning):
        with bp(extension()):
            loss.backward()


@pytest.mark.parametrize("extension", ext_2nd_order, ids=ext_2nd_order_name)
def test_sqrt_hessian_modified_mse_should_fail(extension):
    loss = dummy_mse() * 2

    with pytest.warns(UserWarning):
        with bp(extension()):
            loss.backward()


@pytest.mark.parametrize("extension", ext_2nd_order, ids=ext_2nd_order_name)
def test_sqrt_hessian_mse_on_vectors_should_fail(extension):
    loss = dummy_mse(D=2) * 2

    with pytest.raises(
        RuntimeError, match=r".*MSE between batches of vectors is not implemented yet.*"
    ):
        with bp(extension()):
            loss.backward()
