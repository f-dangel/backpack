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

    def output_formula(W, K, P, S, D):
        try:
            size = (float(W - D * (K - 1) - 1 + 2 * P) // float(S) ) + 1.0
        except ZeroDivisionError:
            size = ((W - K + 2 * P) // S ) + 1
        return size

    K = module.kernel_size[0]
    P = module.padding[0]
    S = module.stride[0]
    out_channels = module.out_channels
    D = module.dilation[0]
    # import pdb; pdb.set_trace()
    if len(input_size[2:]) == 1:
        # Conv1d
        L = input_size[2]
        output_size = output_formula(L, K, P, S, D) * out_channels

    elif len(input_size[2:]) == 2:
        # Conv2d
        H, W = input_size[2], input_size[3]
        output_size = output_formula(H, K, P, S, D) * output_formula(W, K, P, S, D) * out_channels

    elif len(input_size[2:]) == 3:
        # Conv3d
        D, H, W = input_size[2], input_size[3], input_size[4]
        output_size = output_formula(D, K, P, S, D) * output_formula(H, K, P, S, D) * output_formula(W, K, P, S, D) * out_channels
    else:
        ValueError("{}-dimensional Conv. is not implemented.".format(len(input_size[2:])))
    
    return int(output_size)

def make_setting(input_size, conv_class):
    """
    input_size: tuple of input size of (N*C*Image Size)
    input: Activation function & Bias setting
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