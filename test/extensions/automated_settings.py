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
