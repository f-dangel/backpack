from test.core.derivatives.utils import classification_targets

import torch

###
# Helpers
###


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
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 2, 2, bias=bias),
            act_cls(),
            torch.nn.Flatten(),
            torch.nn.Linear(72, 5),
        )

    dict_setting = {
        "input_fn": lambda: torch.rand(3, 3, 7, 7),
        "module_fn": lambda: make_simple_cnn(act_cls, bias),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(),
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
        return torch.nn.Sequential(
            conv_class(*conv_params),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(output_size, 5),
        )

    def get_output_shape(module, module_params, input):
        """Returns the output shape for a given layer."""
        output = module(*module_params)(input)
        return output.numel() // output.shape[0]

    input = torch.rand(input_size)
    output_size = get_output_shape(conv_class, conv_params, input)

    dict_setting = {
        "input_fn": lambda: torch.rand(input_size),
        "module_fn": lambda: make_cnn(conv_class, output_size, conv_params),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
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
        return torch.nn.Sequential(
            conv_class(*conv_params),
            torch.nn.ReLU(),
            pool_cls(*pool_params),
            torch.nn.Flatten(),
            torch.nn.Linear(output_size, 5),
        )

    def get_output_shape(module, module_params, input, pool, pool_params):
        """Returns the output shape for a given layer."""
        output_1 = module(*module_params)(input)
        output = pool_cls(*pool_params)(output_1)
        return output.numel() // output.shape[0]

    conv_params = (3, 2, 2)
    input = torch.rand(input_size)
    output_size = get_output_shape(
        conv_class, conv_params, input, pool_cls, pool_params
    )

    dict_setting = {
        "input_fn": lambda: torch.rand(input_size),
        "module_fn": lambda: make_cnn(
            conv_class, output_size, conv_params, pool_cls, pool_params
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
        "id_prefix": "automated-simple-cnn",
    }

    return dict_setting
