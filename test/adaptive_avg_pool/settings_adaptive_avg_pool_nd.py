"""Settings to run test_adaptive_avg_pool_nd."""
from typing import Any, Dict, List

from torch import Size

SETTINGS: List[Dict[str, Any]] = [
    {
        "N": 1,
        "shape_target": 2,
        "shape_input": (1, 5, 8),
        "works": True,
    },
    {
        "N": 1,
        "shape_target": 2,
        "shape_input": (1, 8, 7),
        "works": False,
    },
    {
        "N": 2,
        "shape_target": Size((4, 3)),
        "shape_input": (1, 64, 8, 9),
        "works": True,
    },
    {
        "N": 2,
        "shape_target": 2,
        "shape_input": (1, 64, 8, 10),
        "works": True,
    },
    {
        "N": 2,
        "shape_target": 2,
        "shape_input": (1, 64, 8, 9),
        "works": False,
    },
    {
        "N": 2,
        "shape_target": (5, 2),
        "shape_input": (1, 64, 64, 10),
        "works": False,
    },
    {
        "N": 3,
        "shape_target": (None, 2, None),
        "shape_input": (1, 64, 7, 10, 5),
        "works": True,
    },
]
