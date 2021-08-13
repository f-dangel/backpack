"""Tests converter.

- whether converted network is equivalent to original network
- whether DiagGGN runs without errors on new network
"""
from test.converter.converter_cases import CONVERTER_MODULES, ConverterModule
from test.utils.skip_test import skip_pytorch_below_1_9_0
from typing import Tuple

from pytest import fixture
from torch import Tensor, allclose, manual_seed, rand_like
from torch.nn import Module, MSELoss

from backpack import backpack, extend
from backpack.extensions import DiagGGNExact


@fixture(
    params=CONVERTER_MODULES,
    ids=[str(model_class) for model_class in CONVERTER_MODULES],
)
def model_and_input(request) -> Tuple[Module, Tensor]:
    """Yield ResNet model and an input to it.

    Args:
        request: pytest request

    Yields:
        model and input
    """
    manual_seed(0)
    skip_pytorch_below_1_9_0()
    model: ConverterModule = request.param()
    inputs: Tensor = model.input_fn()
    inputs.requires_grad = True
    yield model, inputs
    del model


def test_network_diag_ggn(model_and_input):
    """Test whether the given module can compute diag_ggn.

    This test is placed here, because some models are too big to run with PyTorch.
    Thus, a diag_ggn comparison with PyTorch is impossible.
    This test just checks whether it runs on BackPACK without errors.
    Additionally, it checks whether the forward pass is identical to the original model.

    Args:
        model_and_input: module to test
    """
    model_original, x = model_and_input
    result_compare = model_original(x)

    model_extended = extend(model_original, use_converter=True, debug=True)
    result = model_extended(x)

    assert allclose(result, result_compare, atol=1e-3)

    loss = extend(MSELoss())(result, rand_like(result))

    with backpack(DiagGGNExact()):
        loss.backward()
    for name, param in model_extended.named_parameters():
        print(name)
        print(param.grad.shape)
        print(param.diag_ggn_exact.shape)
