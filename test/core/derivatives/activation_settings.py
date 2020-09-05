"""Test configurations for `backpack.core.derivatives` for activation functions


Required entries:
    "module_fn" (callable): Contains a model constructed from `torch.nn` layers
    "input_fn" (callable): Used for specifying input function

Optional entries:
    "target_fn" (callable): Fetches the groundtruth/target classes 
                            of regression/classification task
    "loss_function_fn" (callable): Loss function used in the model
    "device" [list(torch.device)]: List of devices to run the test on.
    "id_prefix" (str): Prefix to be included in the test name.
    "seed" (int): seed for the random number for torch.rand
"""

import torch

ACTIVATION_SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################

example = {
    "module_fn": lambda: torch.nn.LeakyReLU(negative_slope=0.1),
    "input_fn": lambda: torch.rand(size=(5, 5)),
    "device": [torch.device("cpu")],  # optional
    "seed": 0,  # optional
    "id_prefix": "activation-example",  # optional
}
ACTIVATION_SETTINGS.append(example)
###############################################################################
#                                test settings                                #
###############################################################################
ACTIVATION_SETTINGS += [
    {
        "module_fn": lambda: torch.nn.LeakyReLU(negative_slope=0.1),
        "input_fn": lambda: torch.rand(size=(10, 5)),
    },
    {
        "module_fn": lambda: torch.nn.LeakyReLU(negative_slope=0.01),
        "input_fn": lambda: torch.Tensor(
            [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]
        ),
    },
    {
        "module_fn": lambda: torch.nn.LogSigmoid(),
        "input_fn": lambda: torch.rand(size=(10, 5)),
    },
    {
        "module_fn": lambda: torch.nn.LogSigmoid(),
        "input_fn": lambda: torch.Tensor(
            [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]
        ),
    },
    {
        "module_fn": lambda: torch.nn.ELU(alpha=0.3),
        "input_fn": lambda: torch.rand(size=(10, 5)),
    },
    {
        "module_fn": lambda: torch.nn.ELU(alpha=0.01, inplace=False),
        "input_fn": lambda: torch.Tensor(
            [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]
        ),
    },
    {
        "module_fn": lambda: torch.nn.SELU(),
        "input_fn": lambda: torch.rand(size=(10, 5)),
    },
    {
        "module_fn": lambda: torch.nn.SELU(inplace=False),
        "input_fn": lambda: torch.Tensor(
            [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]
        ),
    },
    {
        "module_fn": lambda: torch.nn.ReLU(),
        "input_fn": lambda: torch.rand(size=(2, 10, 5)),
    },
    {
        "module_fn": lambda: torch.nn.ReLU(inplace=False),
        "input_fn": lambda: torch.Tensor(
            [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]
        ),
    },
    {
        "module_fn": lambda: torch.nn.Tanh(),
        "input_fn": lambda: torch.rand(size=(2, 10, 5)),
    },
    {
        "module_fn": lambda: torch.nn.Tanh(),
        "input_fn": lambda: torch.Tensor(
            [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]
        ),
    },
    {
        "module_fn": lambda: torch.nn.Sigmoid(),
        "input_fn": lambda: torch.rand(size=(2, 10, 5)),
    },
    {
        "module_fn": lambda: torch.nn.Sigmoid(),
        "input_fn": lambda: torch.Tensor(
            [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]
        ),
    },
]
