"""Tests converter.

- whether converted network is equivalent to original network
- whether DiagGGN runs without errors on new network
"""
from test.converter.converter_cases import CONVERTER_MODULES, ConverterModule
from test.core.derivatives.utils import classification_targets, regression_targets
from typing import Tuple

from pytest import fixture
from torch import Tensor, allclose, cat, int32, linspace, manual_seed
from torch.nn import CrossEntropyLoss, Module, MSELoss

from backpack import backpack, extend
from backpack.extensions import DiagGGNExact
from backpack.utils.examples import autograd_diag_ggn_exact


@fixture(
    params=CONVERTER_MODULES,
    ids=[str(model_class) for model_class in CONVERTER_MODULES],
)
def model_and_input(request) -> Tuple[Module, Tensor, Module]:
    """Yield ResNet model and an input to it.

    Args:
        request: pytest request

    Yields:
        model and input and loss function
    """
    manual_seed(0)
    model: ConverterModule = request.param()
    inputs: Tensor = model.input_fn()
    loss_fn: Module = model.loss_fn()
    yield model, inputs, loss_fn
    del model, inputs, loss_fn


def test_network_diag_ggn(model_and_input):
    """Test whether the given module can compute diag_ggn.

    This test is placed here, because some models are too big to run with PyTorch.
    Thus, a full diag_ggn comparison with PyTorch is impossible.
    This test just checks whether it runs on BackPACK without errors.
    Additionally, it checks whether the forward pass is identical to the original model.
    Finally, a small number of elements of DiagGGN are compared.

    Args:
        model_and_input: module to test

    Raises:
        NotImplementedError: if loss_fn is not MSELoss or CrossEntropyLoss
    """
    model_original, x, loss_fn = model_and_input
    model_original = model_original.eval()
    output_compare = model_original(x)
    if isinstance(loss_fn, MSELoss):
        y = regression_targets(output_compare.shape)
    elif isinstance(loss_fn, CrossEntropyLoss):
        y = classification_targets(
            (output_compare.shape[0], *output_compare.shape[2:]),
            output_compare.shape[1],
        )
    else:
        raise NotImplementedError(f"test cannot handle loss_fn = {type(loss_fn)}")

    num_params = sum(p.numel() for p in model_original.parameters() if p.requires_grad)
    num_to_compare = 10
    idx_to_compare = linspace(0, num_params - 1, num_to_compare, dtype=int32)
    diag_ggn_exact_to_compare = autograd_diag_ggn_exact(
        x, y, model_original, loss_fn, idx=idx_to_compare
    )

    model_extended = extend(model_original, use_converter=True, debug=True)
    output = model_extended(x)

    assert allclose(output, output_compare)

    loss = extend(loss_fn)(output, y)

    with backpack(DiagGGNExact()):
        loss.backward()

    diag_ggn_exact_vector = cat(
        [
            p.diag_ggn_exact.flatten()
            for p in model_extended.parameters()
            if p.requires_grad
        ]
    )

    for idx, element in zip(idx_to_compare, diag_ggn_exact_to_compare):
        assert allclose(element, diag_ggn_exact_vector[idx], atol=1e-5)
