"""Test configurations for `backpack.core.derivatives`.

Required entries:
    "module_cls" (torch.nn.Module): Module class of layer.
    "module_kwargs" (dict): Dictionary used for layer initialization.
        Leave empty if no arguments are required (e.g. for `torch.nn.ReLU`).
    "in_shape" (tuple): Dimensions of layer input.

Optional entries:
    "device" [list(torch.device)]: List of devices to run the test on.
    "id_prefix" (str): Prefix to be included in the test name.
"""

import torch

SETTINGS = []

example = {
    "module_cls": torch.nn.Linear,
    "module_kwargs": {"in_features": 5, "out_features": 3, "bias": True},
    "in_shape": (10, 5),
    "device": [torch.device("cpu")],
    "id_prefix": "config-example",
}
SETTINGS.append(example)

SETTINGS += [
    {
        "module_cls": torch.nn.Linear,
        "module_kwargs": {"in_features": 7, "out_features": 3, "bias": True,},
        "in_shape": (10, 7,),
    },
    {"module_cls": torch.nn.ReLU, "in_shape": (10, 5)},
    {"module_cls": torch.nn.Tanh, "in_shape": (1, 5, 6),},
    {"module_cls": torch.nn.Sigmoid, "in_shape": (1, 5),},
    {
        "module_cls": torch.nn.Conv2d,
        "module_kwargs": {
            "in_channels": 3,
            "out_channels": 3,
            "kernel_size": 4,
            "stride": 1,
        },
        "in_shape": (1, 3, 32, 32),
    },
    {
        "module_cls": torch.nn.MaxPool2d,
        "module_kwargs": {"kernel_size": 2, "stride": 2},
        "in_shape": (1, 5, 32, 32),
    },
    {
        "module_cls": torch.nn.AvgPool2d,
        "module_kwargs": {"kernel_size": 2},
        "in_shape": (1, 3, 32, 32),
    },
]
