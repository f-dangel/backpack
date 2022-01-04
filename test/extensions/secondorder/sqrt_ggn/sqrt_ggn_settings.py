"""Contains test settings for testing SqrtGGN extension."""
from test.core.derivatives.utils import classification_targets
from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTINGS

from torch import randint
from torch.nn import CrossEntropyLoss, Embedding, Flatten, Linear, Sequential

SQRT_GGN_SETTINGS = SECONDORDER_SETTINGS

###############################################################################
#                               Embedding                                     #
###############################################################################
SQRT_GGN_SETTINGS += [
    {
        "input_fn": lambda: randint(0, 5, (6,)),
        "module_fn": lambda: Sequential(
            Embedding(5, 3),
            Linear(3, 4),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((6,), 4),
    },
    {
        "input_fn": lambda: randint(0, 3, (3, 2, 2)),
        "module_fn": lambda: Sequential(
            Embedding(3, 2),
            Flatten(),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((3,), 2 * 2),
        "seed": 1,
    },
]
