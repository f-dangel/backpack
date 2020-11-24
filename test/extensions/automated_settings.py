import torch
from test.core.derivatives.utils import classification_targets
###
# Helpers
###


def make_simple_cnn_setting(act_cls, bias):
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
        "id_prefix": "automated-simple-cnn",
    }

    return dict_setting

def calculate_output_shape(module, input_size):
    """
    
    input: convolutional module and input shape
    output: shape of the output
    Function to calculate the output size for a convolutional layer

    Formula for output shape:
    Note: A very basic version of the forumla is used.

    1. Convolutional layer:
        `L_out = [(L_in - Dilation * (Kernel - 1) -1 + 2 * Padding) / Stride ] + 1`
    2. Convolutional Transposed layer:
        `L_out = (L_in - 1) * Stride - 2 * Padding + Dilation * (Kernel - 1) + output_padding + 1
    
    """
    K = module.kernel_size[0]
    P = module.padding[0]
    S = module.stride[0]
    out_channels = module.out_channels
    D = module.dilation[0]

    # def output_formula_conv(W, K, P, S, D):
    #     if D == 1:
    #         U = K
    #     else:
    #         U = D * (K - 1) - 1
    #     return (float(W - U + 2 * P) // float(S) ) + 1.0

    def output_formula_conv(W, K, P, S, D):
        # Note: backPack does not support `output_padding` yet
        return (W - 1) * S - 2 * P + D * (K - 1) + 1



    if len(input_size[2:]) == 1:
        # Conv1d
        L_in = input_size[2]
        output_size = output_formula_conv(L_in, K, P, S, D) * out_channels

    elif len(input_size[2:]) == 2:
        # Conv2d
        H_in, W_in = input_size[2], input_size[3]
        output_size = output_formula_conv(H_in, K, P, S, D) * output_formula_conv(W_in, K, P, S, D) * out_channels

    elif len(input_size[2:]) == 3:
        # Conv3d
        D_in, H_in, W_in = input_size[2], input_size[3], input_size[4]
        output_size = output_formula_conv(D_in, K, P, S, D) * output_formula_conv(H_in, K, P, S, D) * output_formula_conv(W_in, K, P, S, D) * out_channels
    else:
        ValueError("{}-dimensional Conv. is not implemented.".format(len(input_size[2:])))
    
    return int(output_size)

def make_setting(input_size, conv_class):
    """
    input_size: tuple of input size of (N*C*Image Size)
    conv_class: convolutional class
    return: simple CNN Network

    This function is used to automatically create a
    simple CNN Network consisting of CNN & Linear layer
    for different activation functions.
    It is used to test `test.extensions`.
    """

    def make_cnn(conv_class, output_size):
        return torch.nn.Sequential(
            conv_class,
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(output_size, 5),
        )
    
    output_size = calculate_output_shape(conv_class, input_size)

    dict_setting = {
        "input_fn": lambda: torch.rand(input_size),
        "module_fn": lambda: make_cnn(conv_class, output_size),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
        "target_fn": lambda: classification_targets((3,), 5),
        "id_prefix": "automated-simple-cnn",
    }

    return dict_setting