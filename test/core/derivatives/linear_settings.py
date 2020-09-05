"""Test configurations for `backpack.core.derivatives` Linear layers


Required entries:
    "module_fn" (torch.nn.Module): Module class of layer.
    "module_kwargs" (dict): Dictionary used for layer initialization.
    "input_kwargs" (dict): Parameters required by torch.nn.Module class

Optional entries:
    "device" [list(torch.device)]: List of devices to run the test on.
    "id_prefix" (str): Prefix to be included in the test name.
"""

import torch

LINEAR_SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################

example = {
    "module_fn": torch.nn.Linear,
    "module_kwargs": {"in_features": 5, "out_features": 3, "bias": True},
    "input_fn": torch.rand,  # optional
    "input_kwargs": {"size": (10, 5)},  # needs to be modified for different "input_fn"
    "device": [torch.device("cpu")],  # optional
    "id_prefix": "layer-example",  # optional
    "seed": 0,  # optional
}
LINEAR_SETTINGS.append(example)

ALLOW_NEW_FORMAT = True
if ALLOW_NEW_FORMAT:
    new_format_example = {
        "module_fn": lambda: torch.nn.Linear(in_features=5, out_features=3, bias=True),
        "input_fn": lambda: torch.rand(size=(10, 5)),
        "target_fn": lambda: None,  # optional
        "device": [torch.device("cpu")],  # optional
        "seed": 0,  # optional
        "id_prefix": "layer-new-format-example",  # optional
    }
    LINEAR_SETTINGS.append(new_format_example)


###############################################################################
#                                test settings                                #
###############################################################################


LINEAR_SETTINGS += [
    {
        "module_fn": torch.nn.Linear,
        "module_kwargs": {
            "in_features": 7,
            "out_features": 3,
            "bias": False,
        },
        "input_kwargs": {"size": (12, 7)},
    },
    {
        "module_fn": torch.nn.Linear,
        "module_kwargs": {
            "in_features": 11,
            "out_features": 10,
            "bias": True,
        },
        "input_kwargs": {"size": (6, 11)},
    },
    {
        "module_fn": torch.nn.Linear,
        "module_kwargs": {
            "in_features": 3,
            "out_features": 2,
            "bias": True,
        },
        "input_fn": torch.Tensor,
        "input_kwargs": {"data": [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]},
    },
]

if ALLOW_NEW_FORMAT:
    LINEAR_SETTINGS += [
        {
            "module_fn": lambda: torch.nn.Linear(
                in_features=7, out_features=3, bias=False
            ),
            "input_fn": lambda: torch.rand(size=(12, 7)),
        },
        {
            "module_fn": lambda: torch.nn.Linear(
                in_features=11, out_features=10, bias=True
            ),
            "input_fn": lambda: torch.rand(size=(6, 11)),
        },
        {
            "module_fn": lambda: torch.nn.Linear(
                in_features=3, out_features=2, bias=True
            ),
            "input_fn": lambda: torch.Tensor(
                [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]
            ),
        },
    ]
