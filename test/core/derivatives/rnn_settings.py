"""Test configurations for `backpack.core.derivatives` RNN layers.

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

RNN_SETTINGS = [
    {
        "module_fn": lambda: torch.nn.RNN(
            input_size=4, hidden_size=3, batch_first=True
        ),
        "input_fn": lambda: torch.rand(size=(3, 5, 4)),
    },
]
