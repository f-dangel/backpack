"""Test automatic support of new layers."""

from test.test___init__ import DEVICES, DEVICES_ID
from typing import Callable, List, Optional, Union

from pytest import mark
from torch import Tensor, allclose, device, manual_seed, rand
from torch.nn import Linear, MSELoss, ReLU, Sequential
from torch.nn.functional import linear, relu

from backpack import backpack, extend, extensions
from backpack.core.derivatives.automatic import AutomaticDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class LinearAutomaticDerivatives(AutomaticDerivatives):
    """Automatic derivatives for ``torch.nn.Linear."""

    @staticmethod
    def as_functional(
        module: Linear,
    ) -> Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]:
        """Return the linear layer's forward pass function.

        Args:
            module: A linear layer.

        Returns:
            The linear layer's forward pass function.
        """
        return linear


class ReLUAutomaticDerivatives(AutomaticDerivatives):
    """Automatic derivatives for ``torch.nn.ReLU."""

    @staticmethod
    def as_functional(module: ReLU) -> Callable[[Tensor], Tensor]:
        """Return the ReLU layer's forward pass function.

        Args:
            module: A ReLU layer.

        Returns:
            The ReLU layer's forward pass function.
        """
        return relu


@mark.parametrize("dev", DEVICES, ids=DEVICES_ID)
def test_automatic_support_diag_ggn_exact(dev: device):
    """Test GGN diagonal computation via automatic derivatives.

    Args:
        dev: The device on which to run the test.
    """

    class DiagGGNExactReLUAutomatic(DiagGGNBaseModule):
        """GGN diagonal computation for ``torch.nn.ReLU`` via automatic derivatives."""

        def __init__(self):
            """Set up the derivatives."""
            super().__init__(ReLUAutomaticDerivatives(), sum_batch=True)

    class DiagGGNExactLinearAutomatic(DiagGGNBaseModule):
        """GGN diag. computation for ``torch.nn.Linear`` via automatic derivatives."""

        def __init__(self):
            """Set up the derivatives."""
            super().__init__(
                LinearAutomaticDerivatives(), params=["weight", "bias"], sum_batch=True
            )

    manual_seed(0)
    X, y = rand(10, 5, device=dev), rand(10, 3, device=dev)

    model = extend(Sequential(Linear(5, 4), ReLU(), Linear(4, 3)).to(dev))
    loss_func = extend(MSELoss().to(dev))

    # ground truth
    with backpack(extensions.DiagGGNExact()):
        loss = loss_func(model(X), y)
        loss.backward()
    diag_ggn = [p.diag_ggn_exact for p in model.parameters()]

    # same quantity with automatic support
    ext = extensions.DiagGGNExact()
    for layer_cls, extension in zip(
        [ReLU, Linear], [DiagGGNExactReLUAutomatic(), DiagGGNExactLinearAutomatic()]
    ):
        ext.set_module_extension(layer_cls, extension, overwrite=True)

    with backpack(ext):
        loss = loss_func(model(X), y)
        loss.backward()
    diag_ggn_automatic = [p.diag_ggn_exact for p in model.parameters()]

    assert len(diag_ggn) == len(diag_ggn_automatic)
    for diag, diag_auto in zip(diag_ggn, diag_ggn_automatic):
        assert allclose(diag, diag_auto)


@mark.parametrize("dev", DEVICES, ids=DEVICES_ID)
def test_automatic_support_batch_diag_ggn_exact(dev: device):
    """Test batch GGN diagonal computation via automatic derivatives.

    Args:
        dev: The device on which to run the test.
    """

    class BatchDiagGGNExactReLUAutomatic(DiagGGNBaseModule):
        """GGN diagonal computation for ``torch.nn.ReLU`` via automatic derivatives."""

        def __init__(self):
            """Set up the derivatives."""
            super().__init__(ReLUAutomaticDerivatives(), sum_batch=False)

    class BatchDiagGGNExactLinearAutomatic(DiagGGNBaseModule):
        """GGN diag. computation for ``torch.nn.Linear`` via automatic derivatives."""

        def __init__(self):
            """Set up the derivatives."""
            super().__init__(
                LinearAutomaticDerivatives(), params=["weight", "bias"], sum_batch=False
            )

    manual_seed(0)
    X, y = rand(10, 5, device=dev), rand(10, 3, device=dev)

    model = extend(Sequential(Linear(5, 4), ReLU(), Linear(4, 3)).to(dev))
    loss_func = extend(MSELoss().to(dev))

    # ground truth
    with backpack(extensions.BatchDiagGGNExact()):
        loss = loss_func(model(X), y)
        loss.backward()
    batch_diag_ggn = [p.diag_ggn_exact_batch for p in model.parameters()]

    # same quantity with automatic support
    ext = extensions.BatchDiagGGNExact()
    for layer_cls, extension in zip(
        [ReLU, Linear],
        [BatchDiagGGNExactReLUAutomatic(), BatchDiagGGNExactLinearAutomatic()],
    ):
        ext.set_module_extension(layer_cls, extension, overwrite=True)

    with backpack(ext):
        loss = loss_func(model(X), y)
        loss.backward()
    batch_diag_ggn_automatic = [p.diag_ggn_exact_batch for p in model.parameters()]

    assert len(batch_diag_ggn) == len(batch_diag_ggn_automatic)
    for batch_diag, batch_diag_auto in zip(batch_diag_ggn, batch_diag_ggn_automatic):
        assert allclose(batch_diag, batch_diag_auto)


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

    class BatchGradLinearAutomatic(BatchGradBase):
        """Batch gradients for ``torch.nn.Linear`` via automatic derivatives."""

        def __init__(self):
            """Set up the derivatives."""
            super().__init__(LinearAutomaticDerivatives(), params=["weight", "bias"])

    manual_seed(0)
    X, y = rand(10, 5, device=dev), rand(10, 3, device=dev)

    model = extend(Sequential(Linear(5, 4), ReLU(), Linear(4, 3)).to(dev))
    loss_func = extend(MSELoss().to(dev))

    # ground truth
    with backpack(extensions.BatchGrad(subsampling=subsampling)):
        loss = loss_func(model(X), y)
        loss.backward()
    batch_grad = [p.grad_batch for p in model.parameters()]

    # same quantity with automatic support
    ext = extensions.BatchGrad(subsampling=subsampling)
    for layer_cls, extension in zip([Linear], [BatchGradLinearAutomatic()]):
        ext.set_module_extension(layer_cls, extension, overwrite=True)

    with backpack(ext):
        loss = loss_func(model(X), y)
        loss.backward()
    batch_grad_automatic = [p.grad_batch for p in model.parameters()]

    assert len(batch_grad) == len(batch_grad_automatic)
    for bg, bg_auto in zip(batch_grad, batch_grad_automatic):
        assert allclose(bg, bg_auto)
