"""Test configurations for `backpack.core.derivatives` for activation functions


Required entries:
    "module_fn" (torch.nn.Module): Module class of layer.
    "module_kwargs" (dict): Dictionary used for layer initialization.
        Leave empty if no arguments are required (e.g. for `torch.nn.ReLU`).
    "input_shape" (tuple): Dimensions of layer input.

Optional entries:
    "device" [list(torch.device)]: List of devices to run the test on.
    "id_prefix" (str): Prefix to be included in the test name.
"""

import torch

activation_SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################

example = {
    "module_fn": torch.nn.LeakyReLU,
    "module_kwargs": {"negative_slope": 0.1},
    "input_fn": torch.rand,  # optional
    "input_kwargs": {
        "size": (5, 5)
    },  # needs to be modified for different "input_fn"; shape atleast n x m
    "device": [torch.device("cpu")],  # optional
    "id_prefix": "act-example",  # optional
    "seed": 0,  # optional
}
activation_SETTINGS.append(example)

###############################################################################
#                                test settings                                #
###############################################################################
activation_SETTINGS += [
    {
        "module_fn": torch.nn.LeakyReLU,
        "module_kwargs": {"negative_slope": 0.1},
        "input_kwargs": {"size": (10, 5)},
    },
    {
        "module_fn": torch.nn.LeakyReLU,
        "module_kwargs": {"negative_slope": 0.01},
        "input_fn": torch.Tensor,
        "input_kwargs": {
            "data": [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]
        },  # Note: Tensor has to be n x m
    },
    {
        "module_fn": torch.nn.LogSigmoid,
        "module_kwargs": {},
        "input_kwargs": {"size": (10, 5)},  # , "data": (-1, 43, 1.2)
    },
    {
        "module_fn": torch.nn.LogSigmoid,
        "module_kwargs": {},
        "input_fn": torch.Tensor,
        "input_kwargs": {
            "data": [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]
        },  # Note: Tensor has to be n x m
    },
    {
        "module_fn": torch.nn.ELU,
        "module_kwargs": {"alpha": 0.3},  # test for different channels
        "input_kwargs": {"size": (10, 5)},  # Note: Tensor has to be n x m
    },
    {
        "module_fn": torch.nn.ELU,
        "module_kwargs": {
            "alpha": 0.1,
            "inplace": False,
        },  # test for different channels
        "input_fn": torch.Tensor,
        "input_kwargs": {
            "data": [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]
        },  # Note: Tensor has to be n x m
    },
    {
        "module_fn": torch.nn.SELU,
        "module_kwargs": {},
        "input_kwargs": {"size": (10, 5, 5)},  # Note: Tensor has to be n x m
    },
    {
        "module_fn": torch.nn.SELU,
        "module_kwargs": {"inplace": False},
        "input_fn": torch.Tensor,
        "input_kwargs": {
            "data": [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]
        },  # Note: Tensor has to be n x m
    },
    {
        "module_fn": torch.nn.ReLU,
        "module_kwargs": {"inplace": False},
        "input_fn": torch.Tensor,
        "input_kwargs": {
            "data": [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]
        },  # Note: Tensor has to be n x m
    },
    {
        "module_fn": torch.nn.ReLU,
        "module_kwargs": {},
        "input_kwargs": {"size": (6, 2, 7)},
    },
    {
        "module_fn": torch.nn.Tanh,
        "module_kwargs": {},
        "input_kwargs": {"size": (1, 5, 6)},
    },
    {
        "module_fn": torch.nn.Tanh,
        "module_kwargs": {},
        "input_fn": torch.Tensor,
        "input_kwargs": {
            "data": [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]
        },  # Note: Tensor has to be n x m
    },
    {
        "module_fn": torch.nn.Sigmoid,
        "module_kwargs": {},
        "input_kwargs": {"size": (1, 5)},
    },
    {
        "module_fn": torch.nn.Sigmoid,
        "module_kwargs": {},
        "input_fn": torch.Tensor,
        "input_kwargs": {
            "data": [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]
        },  # Note: Tensor has to be n x m
    },
]
