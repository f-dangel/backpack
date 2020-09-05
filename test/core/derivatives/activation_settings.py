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

ACTIVATION_SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################

example = {
    "module_fn": torch.nn.LeakyReLU,
    "module_kwargs": {"negative_slope": 0.1},
    "input_fn": torch.rand,  # optional
    "input_kwargs": {"size": (5, 5)},
    # Note: Custom "data" can be added,
    # "data": [(-1, 43, 1.3),..] is used to test for negative cases
    "device": [torch.device("cpu")],  # optional
    "id_prefix": "act-example",  # optional
    "seed": 0,  # optional
}
ACTIVATION_SETTINGS.append(example)

ALLOW_NEW_FORMAT = True
if ALLOW_NEW_FORMAT:
    new_format_example = {
        "module_fn": lambda: torch.nn.LeakyReLU(negative_slope=0.1),
        "input_fn": lambda: torch.rand(size=(5, 5)),
        "device": [torch.device("cpu")],  # optional
        "seed": 0,  # optional
        "id_prefix": "activation-new-format-example",  # optional
    }
ACTIVATION_SETTINGS.append(new_format_example)
###############################################################################
#                                test settings                                #
###############################################################################
if ALLOW_NEW_FORMAT:
    ACTIVATION_SETTINGS += [
    {
        "module_fn": lambda: torch.nn.LeakyReLU(negative_slope=0.1),
        "input_fn": lambda: torch.rand(size=(10, 5)),
    },
    {
        "module_fn": lambda: torch.nn.LeakyReLU(negative_slope=0.01),
        "input_fn": lambda: torch.Tensor([(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]),
    },
    {
        "module_fn": lambda: torch.nn.LogSigmoid(),
        "input_fn": lambda: torch.rand(size=(10, 5)),
    },   
    {
        "module_fn": lambda: torch.nn.LogSigmoid(),
        "input_fn": lambda: torch.Tensor([(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]),
    },
    {
        "module_fn": lambda: torch.nn.ELU(alpha=0.3),
        "input_fn": lambda: torch.rand(size=(10, 5)),
    },
    {
        "module_fn": lambda: torch.nn.ELU(alpha=0.01,inplace=False),
        "input_fn": lambda: torch.Tensor([(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]),
    }, 
    {
        "module_fn": lambda: torch.nn.SELU(),
        "input_fn": lambda: torch.rand(size=(10, 5)),
    },
    {
        "module_fn": lambda: torch.nn.SELU(inplace=False),
        "input_fn": lambda: torch.Tensor([(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]),
    }, 
    {
        "module_fn": lambda: torch.nn.ReLU(),
        "input_fn": lambda: torch.rand(size=(2, 10, 5)),
    },
    {
        "module_fn": lambda: torch.nn.ReLU(inplace=False),
        "input_fn": lambda: torch.Tensor([(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]),
    }, 
    {
        "module_fn": lambda: torch.nn.Tanh(),
        "input_fn": lambda: torch.rand(size=(2, 10, 5)),
    },
    {
        "module_fn": lambda: torch.nn.Tanh(),
        "input_fn": lambda: torch.Tensor([(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]),
    }, 
    {
        "module_fn": lambda: torch.nn.Sigmoid(),
        "input_fn": lambda: torch.rand(size=(2, 10, 5)),
    },
    {
        "module_fn": lambda: torch.nn.Sigmoid(),
        "input_fn": lambda: torch.Tensor([(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]),
    }, 
    ]
ACTIVATION_SETTINGS += [
    {
        "module_fn": torch.nn.LeakyReLU,
        "module_kwargs": {"negative_slope": 0.1},
        "input_kwargs": {"size": (10, 5)},
    },
    {
        "module_fn": torch.nn.LeakyReLU,
        "module_kwargs": {"negative_slope": 0.01},
        "input_fn": torch.Tensor,
        "input_kwargs": {"data": [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]},
    },
    {
        "module_fn": torch.nn.LogSigmoid,
        "module_kwargs": {},
        "input_kwargs": {"size": (10, 5)},
    },
    {
        "module_fn": torch.nn.LogSigmoid,
        "module_kwargs": {},
        "input_fn": torch.Tensor,
        "input_kwargs": {"data": [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]},
    },
    {
        "module_fn": torch.nn.ELU,
        "module_kwargs": {"alpha": 0.3},
        "input_kwargs": {"size": (10, 5)},
    },
    {
        "module_fn": torch.nn.ELU,
        "module_kwargs": {
            "alpha": 0.1,
            "inplace": False,
        },
        "input_fn": torch.Tensor,
        "input_kwargs": {"data": [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]},
    },
    {
        "module_fn": torch.nn.SELU,
        "module_kwargs": {},
        "input_kwargs": {"size": (10, 5, 5)},
    },
    {
        "module_fn": torch.nn.SELU,
        "module_kwargs": {"inplace": False},
        "input_fn": torch.Tensor,
        "input_kwargs": {"data": [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]},
    },
    {
        "module_fn": torch.nn.ReLU,
        "module_kwargs": {"inplace": False},
        "input_fn": torch.Tensor,
        "input_kwargs": {"data": [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]},
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
        "input_kwargs": {"data": [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]},
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
        "input_kwargs": {"data": [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]},
    },
]
