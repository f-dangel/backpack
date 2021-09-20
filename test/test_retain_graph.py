"""Test autograd functionality like retain_graph."""
from pytest import raises
from torch import manual_seed, rand, randint
from torch.nn import CrossEntropyLoss, Linear, Module, Sequential

from backpack import backpack, extend
from backpack.extensions import BatchGrad


def test_retain_graph():
    """Tests whether retain_graph works as expected.

    Does several forward and backward passes.
    In between, it is tested whether BackPACK quantities are present or not.
    """
    manual_seed(0)
    model = extend(Sequential(Linear(4, 6), Linear(6, 5)))
    loss_fn = extend(CrossEntropyLoss())

    # after a forward pass graph is not clear
    loss = loss_fn(model(rand(8, 4)), randint(5, (8,)))
    with raises(AssertionError):
        _check_no_io(model)

    # after a normal backward pass graph should be clear
    loss.backward()
    _check_no_io(model)

    # after a backward pass with retain_graph=True graph is not clear
    loss = loss_fn(model(rand(8, 4)), randint(5, (8,)))
    with backpack(retain_graph=True):
        loss.backward(retain_graph=True)
    with raises(AssertionError):
        _check_no_io(model)

    # doing several backward passes with retain_graph=True
    for _ in range(3):
        with backpack(retain_graph=True):
            loss.backward(retain_graph=True)
    with raises(AssertionError):
        _check_no_io(model)

    # finally doing a normal backward pass that verifies graph is clear again
    with backpack(BatchGrad()):
        loss.backward()
    _check_no_io(model)


def _check_no_io(parent_module: Module) -> None:
    """Checks whether the module is clear of any BackPACK inputs and outputs.

    Args:
        parent_module: The module to test

    Raises:
        AssertionError: if the module or any child module has BackPACK inputs or outputs.
    """
    for module in parent_module.children():
        _check_no_io(module)
    if hasattr(parent_module, "input0") or hasattr(parent_module, "output"):
        raise AssertionError(
            f"graph should be clear, but {parent_module} has input0 or output."
        )
