"""Test whether torchvision is extendable with graph utils."""
from test.utils.skip_test import skip_pytorch_below_1_9_0
from typing import Tuple

import torchvision.models
from pytest import fixture
from torch import Tensor, allclose, manual_seed, rand, rand_like
from torch.nn import Module, MSELoss

from backpack import backpack, extend
from backpack.extensions import DiagGGNExact


@fixture
def model_and_input() -> Tuple[Module, Tensor]:
    """Yield ResNet model and an input to it.

    Yields:
        model and input
    """
    manual_seed(0)
    skip_pytorch_below_1_9_0()
    resnet18 = torchvision.models.resnet18(num_classes=4).eval()
    yield resnet18, rand(2, 3, 7, 7, requires_grad=True)
    del resnet18


def test_network_diag_ggn(model_and_input):
    """Test whether the given module can compute diag_ggn.

    Args:
        model_and_input: module to test
    """
    model_original, x = model_and_input
    result_compare = model_original(x)

    model_extended = extend(model_original, use_converter=True)
    result = model_extended(x)

    assert allclose(result, result_compare, atol=1e-3)

    loss = extend(MSELoss())(result, rand_like(result))

    with backpack(DiagGGNExact()):
        loss.backward()
    for name, param in model_extended.named_parameters():
        print(name)
        print(param.grad.shape)
        print(param.diag_ggn_exact.shape)
