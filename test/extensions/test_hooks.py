"""Tests for extensions hook.

These tests aim at demonstrating the pitfalls one may run into when using hooks that
iterate over ``module.parameters()``.
"""
from test.core.derivatives.utils import classification_targets, get_available_devices
from typing import Tuple

from pytest import fixture, mark, raises
from torch import Tensor, manual_seed, rand
from torch.nn import CrossEntropyLoss, Linear, Module, Sequential

from backpack import backpack, extend
from backpack.extensions import BatchGrad, DiagGGNExact
from backpack.extensions.backprop_extension import FAIL_ERROR, BackpropExtension

DEVICES = get_available_devices()
DEVICES_ID = [str(dev) for dev in DEVICES]

NESTED_SEQUENTIAL = "NESTED_SEQUENTIAL"
CUSTOM_CONTAINER = "CUSTOM_CONTAINER"
problem_list = [NESTED_SEQUENTIAL, CUSTOM_CONTAINER]


@fixture(params=DEVICES, ids=DEVICES_ID)
def device(request):
    """Yields the available device for the test.

    Args:
        request: pytest request

    Yields:
        an available device
    """
    yield request.param


@fixture(params=problem_list, ids=problem_list)
def problem(device, request) -> Tuple[Module, Tensor, str]:
    """Return extended nested sequential with loss from a forward pass.

    Args:
        device: available device
        request: pytest request

    Yields:
        model, loss and problem_string

    Raises:
        NotImplementedError: if the problem_string is unknown
    """
    problem_string = request.param
    manual_seed(0)

    B = 2
    X = rand(B, 4).to(device)
    y = classification_targets((B,), 2).to(device)

    if problem_string == NESTED_SEQUENTIAL:
        model = Sequential(
            Linear(4, 3, bias=False),
            Sequential(
                Linear(3, 2, bias=False),
            ),
        )
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

        model = _MyCustomModule()
    else:
        raise NotImplementedError(
            f"problem={problem_string} but no test setting for this."
        )

    model = extend(model.to(device))
    lossfunc = extend(CrossEntropyLoss(reduction="mean").to(device))
    loss = lossfunc(model(X), y)
    yield model, loss, problem_string


@mark.parametrize(
    "extension", [BatchGrad(), DiagGGNExact()], ids=["BatchGrad", "DiagGGNExact"]
)
def test_extension_hook_multiple_parameter_visits(
    problem, extension: BackpropExtension
):
    """Tests whether each parameter is visited exactly once.

    For those cases where parameters are visited more than once (e.g. Custom containers),
    it tests that an error is raised.

    Furthermore, it is tested whether first order extensions run fine in either case,
    and second order extensions raise an error in the case of custom containers.

    Args:
        problem: test problem, consisting of model, loss, and problem_string
        extension: first or second order extension to test

    Raises:
        NotImplementedError: if the problem_string is unknown
    """
    model, loss, problem_string = problem

    params_visited = {id(p): 0 for p in model.parameters()}

    def count_visits(module):
        """Increase counter in ``params_visited`` for all parameters in ``module``.

        Args:
            module: the module of which the parameter visits are counted
        """
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
        """Checks whether all parameters have been visited exactly once.

        Raises:
            AssertionError: if a parameter hasn't been visited exactly once
        """
        for param_id, visits in params_visited.items():
            if visits != 1:
                raise AssertionError(f"Hook visited param {param_id} {visits}≠1 times")

    if problem_string == NESTED_SEQUENTIAL:
        check_all_parameters_visited_once()
    elif problem_string == CUSTOM_CONTAINER:
        with raises(AssertionError):
            check_all_parameters_visited_once()
    else:
        raise NotImplementedError(f"unknown problem_string={problem_string}")


def test_extension_hook_param_before_savefield_exists(problem):
    """Extension hooks iterating over parameters may get called before BackPACK.

    This leads to the case, that the BackPACK quantities might not be calculated yet.
    Thus, derived quantities cannot be calculated.

    Sequential containers just work fine.
    Custom containers crash.

    Args:
        problem: problem consisting of model, loss, and problem_string

    Raises:
        NotImplementedError: if problem_string is unknown
    """
    _, loss, problem_string = problem

    params_without_grad_batch = []

    def check_grad_batch(module):
        """Check whether the module has a grad_batch attribute.

        Args:
            module: the module to check

        Raises:
            AssertionError: if a parameter does not have grad_batch attribute.
        """
        for p in module.parameters():
            if not hasattr(p, "grad_batch"):
                params_without_grad_batch.append(id(p))
                raise AssertionError(f"Param {id(p)} has no 'grad_batch' attribute")

    if problem_string == NESTED_SEQUENTIAL:
        with backpack(BatchGrad(), extension_hook=check_grad_batch, debug=True):
            loss.backward()

        assert len(params_without_grad_batch) == 0
    elif problem_string == CUSTOM_CONTAINER:
        with raises(AssertionError):
            with backpack(BatchGrad(), extension_hook=check_grad_batch, debug=True):
                loss.backward()
        assert len(params_without_grad_batch) > 0
    else:
        raise NotImplementedError(f"unknown problem_string={problem_string}")
