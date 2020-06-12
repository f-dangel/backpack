from test.automated_test import check_sizes_and_values
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.problem import ExtensionsTestProblem

import pytest
import torch

dummy_setting = {
    "input_fn": lambda: None,
    "module_fn": lambda: None,
    "loss_function_fn": lambda: None,
    "target_fn": lambda: None,
    "device": torch.device("cpu"),
    "seed": 0,
    "id_prefix": "",
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
