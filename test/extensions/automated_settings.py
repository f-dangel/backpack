from test.core.derivatives.utils import classification_targets

from torch import rand
from torch.nn import Conv2d, CrossEntropyLoss, Flatten, Linear, Module, ReLU, Sequential

###
# Helpers
###


def set_requires_grad(model: Module, new_requires_grad) -> None:
    """Set the ``requires_grad`` attribute of the module parameters.

    Args:
        module: Network or layer.
        new_requires_grad: New value for ``requires_grad``.
    """
    for p in model.parameters():
        p.requires_grad = new_requires_grad


def make_simple_act_setting(act_cls, bias):
    """
    input: Activation function & Bias setting
    return: simple CNN Network

    This function is used to automatically create a
    simple CNN Network consisting of CNN & Linear layer
    for different activation functions.
    It is used to test `test.extensions`.
    """

    def make_simple_cnn(act_cls, bias):
        linear = Linear(72, 5)
        set_requires_grad(linear, False)

        return Sequential(Conv2d(3, 2, 2, bias=bias), act_cls(), Flatten(), linear)

    dict_setting = {
        "input_fn": lambda: rand(3, 3, 7, 7),
        "module_fn": lambda: make_simple_cnn(act_cls, bias),
        "loss_function_fn": lambda: CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((3,), 5),
        "id_prefix": "automated-simple-cnn-act",
    }

    return dict_setting


def make_simple_cnn_setting(input_size, conv_class, conv_params):
    """
    input_size: tuple of input size of (N*C*Image Size)
    conv_class: convolutional class
    conv_params: configurations for convolutional class
    return: simple CNN Network

    This function is used to automatically create a
    simple CNN Network consisting of CNN & Linear layer
    for different convolutional layers.
    It is used to test `test.extensions`.
    """

    def make_cnn(conv_class, output_size, conv_params):
        """Note: output class size is assumed to be 5"""
        linear = Linear(output_size, 5)
        set_requires_grad(linear, False)

        return Sequential(conv_class(*conv_params), ReLU(), Flatten(), linear)

    def get_output_shape(module, module_params, input):
        """Returns the output shape for a given layer."""
        output = module(*module_params)(input)
        return output.numel() // output.shape[0]

    input = rand(input_size)
    output_size = get_output_shape(conv_class, conv_params, input)

    dict_setting = {
        "input_fn": lambda: rand(input_size),
        "module_fn": lambda: make_cnn(conv_class, output_size, conv_params),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
        "id_prefix": "automated-simple-cnn",
    }

    return dict_setting


def make_simple_pooling_setting(input_size, conv_class, pool_cls, pool_params):
    """
    input_size: tuple of input size of (N*C*Image Size)
    conv_class: convolutional class
    conv_params: configurations for convolutional class
    return: simple CNN Network

    This function is used to automatically create a
    simple CNN Network consisting of CNN & Linear layer
    for different convolutional layers.
    It is used to test `test.extensions`.
    """

    def make_cnn(conv_class, output_size, conv_params, pool_cls, pool_params):
        """Note: output class size is assumed to be 5"""
        linear = Linear(output_size, 5)
        set_requires_grad(linear, False)

        return Sequential(
            conv_class(*conv_params), ReLU(), pool_cls(*pool_params), Flatten(), linear
        )

    def get_output_shape(module, module_params, input, pool, pool_params):
        """Returns the output shape for a given layer."""
        output_1 = module(*module_params)(input)
        output = pool_cls(*pool_params)(output_1)
        return output.numel() // output.shape[0]

    conv_params = (3, 2, 2)
    input = rand(input_size)
    output_size = get_output_shape(
        conv_class, conv_params, input, pool_cls, pool_params
    )

    dict_setting = {
        "input_fn": lambda: rand(input_size),
        "module_fn": lambda: make_cnn(
            conv_class, output_size, conv_params, pool_cls, pool_params
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
        "id_prefix": "automated-simple-cnn",
    }

    return dict_setting
