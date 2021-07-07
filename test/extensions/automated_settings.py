"""Contains helpers to create CNN test cases."""
from test.core.derivatives.utils import classification_targets
from typing import Any, Tuple, Type

from torch import Tensor, rand
from torch.nn import Conv2d, CrossEntropyLoss, Flatten, Linear, Module, ReLU, Sequential


def set_requires_grad(model: Module, new_requires_grad: bool) -> None:
    """Set the ``requires_grad`` attribute of the model parameters.

    Args:
        model: Network or layer.
        new_requires_grad: New value for ``requires_grad``.
    """
    for p in model.parameters():
        p.requires_grad = new_requires_grad


def make_simple_act_setting(act_cls: Type[Module], bias: bool) -> dict:
    """Create a simple CNN with activation as test case dictionary.

    Make parameters of final linear layer non-differentiable to save run time.

    Args:
        act_cls: Class of the activation function.
        bias: Use bias in the convolution.

    Returns:
        Dictionary representation of the simple CNN test case.
    """

    def _make_simple_cnn(act_cls: Type[Module], bias: bool) -> Sequential:
        linear = Linear(72, 5)
        set_requires_grad(linear, False)

        return Sequential(Conv2d(3, 2, 2, bias=bias), act_cls(), Flatten(), linear)

    dict_setting = {
        "input_fn": lambda: rand(3, 3, 7, 7),
        "module_fn": lambda: _make_simple_cnn(act_cls, bias),
        "loss_function_fn": lambda: CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((3,), 5),
        "id_prefix": "automated-simple-cnn-act",
    }

    return dict_setting


def make_simple_cnn_setting(
    input_size: Tuple[int], conv_cls: Type[Module], conv_params: Tuple[Any]
) -> dict:
    """Create ReLU CNN with convolution hyperparameters as test case dictionary.

    Make parameters of final linear layer non-differentiable to save run time.

    Args:
        input_size: Input shape ``[N, C_in, ...]``.
        conv_cls: Class of convolution layer.
        conv_params: Convolution hyperparameters.

    Returns:
        Dictionary representation of the test case.
    """

    def _make_cnn(
        conv_cls: Type[Module], output_dim: int, conv_params: Tuple
    ) -> Sequential:
        linear = Linear(output_dim, 5)
        set_requires_grad(linear, False)

        return Sequential(conv_cls(*conv_params), ReLU(), Flatten(), linear)

    input = rand(input_size)
    output_dim = _get_output_dim(conv_cls(*conv_params), input)

    dict_setting = {
        "input_fn": lambda: rand(input_size),
        "module_fn": lambda: _make_cnn(conv_cls, output_dim, conv_params),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
        "id_prefix": "automated-simple-cnn",
    }

    return dict_setting


def make_simple_pooling_setting(
    input_size: Tuple[int],
    conv_cls: Type[Module],
    pool_cls: Type[Module],
    pool_params: Tuple[Any],
) -> dict:
    """Create CNN with convolution and pooling layer as test case dictionary.

    Make parameters of final linear layer non-differentiable to save run time.

    Args:
        input_size: Input shape ``[N, C_in, ...]``.
        conv_cls: Class of convolution layer.
        pool_cls: Class of pooling layer.
        pool_params: Pooling hyperparameters.

    Returns:
        Dictionary representation of the test case.
    """

    def _make_cnn(
        conv_cls: Type[Module],
        output_size: int,
        conv_params: Tuple[Any],
        pool_cls: Type[Module],
        pool_params: Tuple[Any],
    ) -> Sequential:
        linear = Linear(output_size, 5)
        set_requires_grad(linear, False)

        return Sequential(
            conv_cls(*conv_params), ReLU(), pool_cls(*pool_params), Flatten(), linear
        )

    conv_params = (3, 2, 2)
    input = rand(input_size)
    output_dim = _get_output_dim(
        Sequential(conv_cls(*conv_params), pool_cls(*pool_params)), input
    )

    dict_setting = {
        "input_fn": lambda: rand(input_size),
        "module_fn": lambda: _make_cnn(
            conv_cls, output_dim, conv_params, pool_cls, pool_params
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
        "id_prefix": "automated-simple-cnn",
    }

    return dict_setting


def _get_output_dim(module: Module, input: Tensor) -> int:
    output = module(input)
    return output.numel() // output.shape[0]
