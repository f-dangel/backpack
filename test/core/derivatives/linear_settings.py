"""Test configurations for `backpack.core.derivatives` Linear layers


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

LINEAR_SETTINGS = []

###############################################################################
#                                   examples                                  #
###############################################################################

example = {
    "module_fn": lambda: torch.nn.Linear(in_features=5, out_features=3, bias=True),
    "input_fn": lambda: torch.rand(size=(10, 5)),
    "target_fn": lambda: None,  # optional
    "device": [torch.device("cpu")],  # optional
    "seed": 0,  # optional
    "id_prefix": "layer-example",  # optional
}
LINEAR_SETTINGS.append(example)


###############################################################################
#                                test settings                                #
###############################################################################

LINEAR_SETTINGS += [
    {
        "module_fn": lambda: torch.nn.Linear(in_features=7, out_features=3, bias=False),
        "input_fn": lambda: torch.rand(size=(12, 7)),
    },
    {
        "module_fn": lambda: torch.nn.Linear(
            in_features=11, out_features=10, bias=True
        ),
        "input_fn": lambda: torch.rand(size=(6, 11)),
    },
    {
        "module_fn": lambda: torch.nn.Linear(in_features=3, out_features=2, bias=True),
        "input_fn": lambda: torch.Tensor(
            [(-1, 43, 1.3), (-2, -0.3, 2.3), (0, -4, 0.33)]
        ),
    },
]
