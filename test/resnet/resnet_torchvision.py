"""Test whether torchvision is extendable with graph utils."""
from test.utils.skip_test import skip_pytorch_below_1_9_0
from typing import Tuple

import pytest
import torchvision.models
from torch import Tensor, rand, rand_like
from torch.nn import Module, MSELoss

from backpack import backpack, extend
from backpack.extensions import DiagGGNExact


@pytest.fixture
def model_and_input() -> Tuple[Module, Tensor]:
    """Yield ResNet model and an input to it.

    Yields:
        model and input
    """
    skip_pytorch_below_1_9_0()
    resnet18 = torchvision.models.resnet18(num_classes=4).eval()
    yield extend(resnet18, use_converter=True), rand(2, 3, 7, 7, requires_grad=True)
    del resnet18


def test_network_diag_ggn(model_and_input):
    """Test whether the given module can compute diag_ggn.

    Args:
        model_and_input: module to test
    """
    model, x = model_and_input

    result = model(x)
    loss = extend(MSELoss())(result, rand_like(result))

    with backpack(DiagGGNExact()):
        loss.backward()
    for name, param in model.named_parameters():
        print(name)
        print(param.grad.shape)
        print(param.diag_ggn_exact.shape)
