"""Tests for extensions hook.

These tests aim at demonstrating the pitfalls one may run into when using hooks that
iterate over ``module.parameters()``.
"""
from test.core.derivatives.utils import classification_targets, get_available_devices

import torch
from pytest import fixture, mark, raises, skip
from torch.nn import Linear, Module, Sequential

from backpack import backpack, extend, extensions
from backpack.extensions import BatchGrad, DiagGGNExact
from backpack.extensions.backprop_extension import FAIL_ERROR, BackpropExtension

DEVICES = get_available_devices()
DEVICES_ID = [str(dev) for dev in DEVICES]

NESTED_SEQUENTIAL = "NESTED_SEQUENTIAL"
CUSTOM_CONTAINER = "CUSTOM_CONTAINER"
problem_list = [NESTED_SEQUENTIAL, CUSTOM_CONTAINER]


@fixture(params=DEVICES, ids=DEVICES_ID)
def device(request):
    yield request.param


@fixture(params=problem_list, ids=problem_list)
def problem(device, request):
    """Return extended nested sequential with loss from a forward pass."""
    problem_string = request.param
    torch.manual_seed(0)

    if problem_string == NESTED_SEQUENTIAL:
        B = 2
        X = torch.rand(B, 4).to(device)
        y = classification_targets((B,), 2).to(device)

        model = Sequential(
            Linear(4, 3, bias=False),
            Sequential(
                Linear(3, 2, bias=False),
            ),
        ).to(device)
    elif problem_string == CUSTOM_CONTAINER:

        class _MyCustomModule(Module):
            def __init__(self):
                super().__init__()
                self.linear1 = Linear(4, 3, bias=False)
                self.linear2 = Linear(3, 2, bias=False)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        B = 2
        X = torch.rand(B, 4).to(device)
        y = classification_targets((B,), 2).to(device)

        model = _MyCustomModule().to(device)
    else:
        raise NotImplementedError(
            f"problem={problem_string} but no test setting for this."
        )

    model = extend(model)
    lossfunc = extend(torch.nn.CrossEntropyLoss(reduction="mean"))
    loss = lossfunc(model(X), y)
    yield model, loss, problem_string


@mark.parametrize(
    "extension", [BatchGrad(), DiagGGNExact()], ids=["BatchGrad", "DiagGGNExact"]
)
def test_extension_hook_multiple_parameter_visits(
    problem, extension: BackpropExtension
):
    """Extension hooks iterating over parameters may traverse them more than once."""
    model, loss, problem_string = problem

    params_visited = {id(p): 0 for p in model.parameters()}

    def count_visits(module):
        """Increase counter in ``params_visited`` for all parameters in ``module``."""
        for p in module.parameters():
            params_visited[id(p)] += 1

    if problem_string == CUSTOM_CONTAINER and extension._fail_mode == FAIL_ERROR:
        with raises(NotImplementedError):
            with backpack(extension, extension_hook=count_visits, debug=True):
                loss.backward()
        return
    with backpack(extension, extension_hook=count_visits, debug=True):
        loss.backward()

    def check_all_parameters_visited_once():
        """Raise ``AssertionError`` if a parameter has been visited more than once."""
        for param_id, visits in params_visited.items():
            if visits == 0:
                raise ValueError(f"Hook never visited param {param_id}")
            elif visits == 1:
                pass
            else:
                raise AssertionError(f"Hook visited param {param_id} {visits} times ")

    if problem_string == NESTED_SEQUENTIAL:
        check_all_parameters_visited_once()
    elif problem_string == CUSTOM_CONTAINER:
        with raises(AssertionError):
            check_all_parameters_visited_once()
    else:
        raise NotImplementedError(f"unknown problem_string={problem_string}")


def test_extension_hook_param_before_savefield_exists(problem):
    """Extension hooks iterating over parameters may get called before BackPACK."""
    _, loss, problem_string = problem

    params_without_grad_batch = []

    def check_grad_batch(module):
        """Raise ``AssertionError`` if one parameter misses ``'grad_batch'``."""
        for p in module.parameters():
            if not hasattr(p, "grad_batch"):
                params_without_grad_batch.append(id(p))
                raise AssertionError(f"Param {id(p)} has no 'grad_batch' attribute")

    if problem_string == NESTED_SEQUENTIAL:
        with backpack(
            extensions.BatchGrad(), extension_hook=check_grad_batch, debug=True
        ):
            loss.backward()

        assert len(params_without_grad_batch) == 0
    elif problem_string == CUSTOM_CONTAINER:
        with raises(AssertionError):
            with backpack(
                extensions.BatchGrad(), extension_hook=check_grad_batch, debug=True
            ):
                loss.backward()
        assert len(params_without_grad_batch) > 0
    else:
        raise NotImplementedError(f"unknown problem_string={problem_string}")
