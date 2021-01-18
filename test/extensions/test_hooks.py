"""Tests for extensions hook.

These tests aim at demonstrating the pitfalls one may run into when using hooks that
iterate over ``module.parameters()``.
"""

from test.core.derivatives.utils import classification_targets, get_available_devices

import pytest
import torch

from backpack import backpack, extend, extensions

DEVICES = get_available_devices()
DEVICES_ID = [str(dev) for dev in DEVICES]


def set_up(device):
    """Return extended nested sequential with loss from a forward pass."""
    torch.manual_seed(0)

    B = 2
    X = torch.rand(B, 4).to(device)
    y = classification_targets((B,), 2).to(device)

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3, bias=False),
        torch.nn.Sequential(
            torch.nn.Linear(3, 2, bias=False),
        ),
    ).to(device)

    model = extend(model)
    lossfunc = extend(torch.nn.CrossEntropyLoss(reduction="mean"))

    loss = lossfunc(model(X), y)

    return model, loss


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
def test_extension_hook_multiple_parameter_visits(device):
    """Extension hooks iterating over parameters may traverse them more than once."""
    model, loss = set_up(device)

    params_visited = {id(p): 0 for p in model.parameters()}

    def count_visits(module):
        """Increase counter in ``params_visited`` for all parameters in ``module``."""
        for p in module.parameters():
            params_visited[id(p)] += 1

    with backpack(extension_hook=count_visits, debug=True):
        loss.backward()

    def check():
        """Raise ``AssertionError`` if a parameter has been visited more than once."""
        for param_id, visits in params_visited.items():
            if visits == 0:
                raise ValueError(f"Hook never visited param {param_id}")
            elif visits == 1:
                pass
            else:
                raise AssertionError(f"Hook visited param {param_id} {visits} times ")

    with pytest.raises(AssertionError):
        check()


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
def test_extension_hook_param_before_savefield_exists(device):
    """Extension hooks iterating over parameters may get called before BackPACK."""
    _, loss = set_up(device)

    params_without_grad_batch = []

    def check_grad_batch(module):
        """Raise ``AssertionError`` if one parameter misses ``'grad_batch'``."""
        for p in module.parameters():
            if not hasattr(p, "grad_batch"):
                params_without_grad_batch.append(id(p))
                raise AssertionError(f"Param {id(p)} has no 'grad_batch' attribute")

    # AssertionError is caught inside BackPACK and will raise a RuntimeError
    with pytest.raises(RuntimeError):
        with backpack(
            extensions.BatchGrad(), extension_hook=check_grad_batch, debug=True
        ):
            loss.backward()

    assert len(params_without_grad_batch) > 0
