"""Test automatic support of new layers."""

from test.automatic_extensions import (
    BatchDiagGGNExactLinearAutomatic,
    BatchDiagGGNExactReLUAutomatic,
    BatchDiagGGNExactSigmoidAutomatic,
    BatchGradLinearAutomatic,
    DiagGGNExactLinearAutomatic,
    DiagGGNExactReLUAutomatic,
    DiagGGNExactSigmoidAutomatic,
)
from test.test___init__ import DEVICES, DEVICES_ID
from test.utils import popattr
from typing import List, Union

from pytest import mark, raises
from torch import allclose, device, manual_seed, rand
from torch.nn import Linear, MSELoss, ReLU, Sequential, Sigmoid

from backpack import backpack, extend, extensions


@mark.parametrize("batched", [False, True], ids=["DiagGGNExact", "BatchDiagGGNExact"])
@mark.parametrize("dev", DEVICES, ids=DEVICES_ID)
def test_automatic_support_diag_ggn_exact(dev: device, batched: bool):
    """Test GGN diagonal computation via automatic derivatives.

    Args:
        dev: The device on which to run the test.
        batched: Whether to compute the batched or summed GGN diagonal.
    """
    manual_seed(0)
    X, y = rand(10, 5, device=dev), rand(10, 3, device=dev)

    model = extend(Sequential(Linear(5, 4), ReLU(), Linear(4, 3), Sigmoid()).to(dev))
    loss_func = extend(MSELoss().to(dev))
    savefield = "diag_ggn_exact_batch" if batched else "diag_ggn_exact"

    # ground truth
    ext = extensions.BatchDiagGGNExact() if batched else extensions.DiagGGNExact()
    with backpack(ext):
        loss = loss_func(model(X), y)
        loss.backward()
    manual = [popattr(p, savefield) for p in model.parameters()]

    # same quantity with automatic support
    ext = extensions.BatchDiagGGNExact() if batched else extensions.DiagGGNExact()
    new_mappings = {
        ReLU: (
            BatchDiagGGNExactReLUAutomatic() if batched else DiagGGNExactReLUAutomatic()
        ),
        Linear: (
            BatchDiagGGNExactLinearAutomatic()
            if batched
            else DiagGGNExactLinearAutomatic()
        ),
        Sigmoid: (
            BatchDiagGGNExactSigmoidAutomatic()
            if batched
            else DiagGGNExactSigmoidAutomatic()
        ),
    }
    for layer_cls, extension in new_mappings.items():
        # make sure we need to turn on explicit overwriting
        with raises(ValueError):
            ext.set_module_extension(layer_cls, extension)
        ext.set_module_extension(layer_cls, extension, overwrite=True)

    with backpack(ext):
        loss = loss_func(model(X), y)
        loss.backward()
    automatic = [popattr(p, savefield) for p in model.parameters()]

    assert len(manual) == len(automatic)
    for m, a in zip(manual, automatic):
        assert allclose(m, a)


SUBSAMPLINGS = [None, [7, 2, 4]]
SUBSAMPLING_IDS = [f"subsampling={subsampling}" for subsampling in SUBSAMPLINGS]


@mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLING_IDS)
@mark.parametrize("dev", DEVICES, ids=DEVICES_ID)
def test_automatic_support_batch_grad(dev: device, subsampling: Union[None, List[int]]):
    """Test per-example gradient computation via automatic derivatives.

    Args:
        dev: The device on which to run the test.
        subsampling: Indices of active samples. ``None`` means full batch.
    """
    manual_seed(0)
    X, y = rand(10, 5, device=dev), rand(10, 3, device=dev)

    model = extend(Sequential(Linear(5, 4), ReLU(), Linear(4, 3), Sigmoid()).to(dev))
    loss_func = extend(MSELoss().to(dev))
    savefield = "grad_batch"

    # ground truth
    with backpack(extensions.BatchGrad(subsampling=subsampling)):
        loss = loss_func(model(X), y)
        loss.backward()
    manual = [popattr(p, savefield) for p in model.parameters()]

    # same quantity with automatic support
    ext = extensions.BatchGrad(subsampling=subsampling)
    new_mappings = {Linear: BatchGradLinearAutomatic()}
    for layer_cls, extension in new_mappings.items():
        # make sure we need to turn on explicit overwriting
        with raises(ValueError):
            ext.set_module_extension(layer_cls, extension)
        ext.set_module_extension(layer_cls, extension, overwrite=True)

    with backpack(ext):
        loss = loss_func(model(X), y)
        loss.backward()
    automatic = [popattr(p, savefield) for p in model.parameters()]

    assert len(manual) == len(automatic)
    for m, a in zip(manual, automatic):
        assert allclose(m, a)
