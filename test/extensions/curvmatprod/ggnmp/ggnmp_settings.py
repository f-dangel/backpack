"""Test cases for the block diagonal matrix-free GGN-matrix product (``GGNMP``).

Cases are shared with the curvature matrix cases in
``test.extensions.curvmatprod.curvmatprod_settings`` and additional cases
can be added here by appending to ``LOCAL_SETTINGS``.
"""

from test.core.derivatives.utils import classification_targets
from test.extensions.curvmatprod.curvmatprod_settings import CURVMATPROD_SETTINGS

from torch import rand
from torch.nn import BatchNorm1d, CrossEntropyLoss, Linear, ReLU, Sequential

SHARED_SETTINGS = CURVMATPROD_SETTINGS
LOCAL_SETTINGS = [
    {
        "input_fn": lambda: rand(3, 10),
        "module_fn": lambda: Sequential(
            Linear(10, 7), BatchNorm1d(7), ReLU(), Linear(7, 5), BatchNorm1d(5)
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((3,), 5),
        "id_prefix": "batch-norm",
    },
]
GGNMP_SETTINGS = SHARED_SETTINGS + LOCAL_SETTINGS
