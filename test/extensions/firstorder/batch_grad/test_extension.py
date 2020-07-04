from test.automated_test import check_sizes_and_values
from test.core.derivatives.utils import classification_targets
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.problem import ExtensionsTestProblem

import pytest
import torch

dummy_setting = {
    "input_fn": lambda: torch.rand(3, 10),
    "module_fn": lambda: torch.nn.Sequential(
        torch.nn.Linear(10, 7), torch.nn.Sigmoid(), torch.nn.Linear(7, 5)
    ),
    "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
    "target_fn": lambda: classification_targets((3,), 5),
    "device": torch.device("cpu"),
    "seed": 0,
    "id_prefix": "prototype",
}


problem = ExtensionsTestProblem(**dummy_setting)

PROBLEMS = [problem]
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_batch_grad(problem):
    """Test individual gradients

    Args:
        problem (ExtensionsTestProblem): Problem for extension test.
    """
    problem.set_up()

    backpack_res = BackpackExtensions(problem).batch_grad()
    autograd_res = AutogradExtensions(problem).batch_grad()

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()
