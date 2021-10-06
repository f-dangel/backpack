"""Test autograd functionality like retain_graph."""
from test.automated_test import check_sizes_and_values

from pytest import raises
from torch import autograd, manual_seed, ones_like, rand, randint, randn, zeros
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
    inputs = rand(8, 4)
    labels = randint(5, (8,))
    loss = loss_fn(model(inputs), labels)
    with raises(AssertionError):
        _check_no_io(model)

    # after a normal backward pass graph should be clear
    loss.backward()
    _check_no_io(model)

    # after a backward pass with retain_graph=True graph is not clear
    loss = loss_fn(model(inputs), labels)
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


def _check_no_io(module: Module) -> None:
    """Checks whether the module is clear of any BackPACK inputs and outputs.

    Args:
        module: The module to test

    Raises:
        AssertionError: if the module or any child module has BackPACK inputs or outputs.
    """
    for child_module in module.children():
        _check_no_io(child_module)

    io_strs = ["input0", "output"]
    if any(hasattr(module, io) for io in io_strs):
        raise AssertionError(f"IO should be clear, but {module} has one of {io_strs}.")


def test_for_loop_replace() -> None:
    """Application of retain_graph: replace an outer for-loop.

    This test is based on issue #220 opened by Romain3Ch216.
    It computes per-component individual gradients of a tensor-valued output
    with a for loop over components, rather than over samples and components.
    """
    manual_seed(0)
    B = 5
    M = 3
    h = 2

    x = randn(B, h)
    fc = extend(Linear(h, M))
    A = fc(x)

    grad_autograd = zeros(B, M, *fc.weight.shape)
    for b in range(B):
        for m in range(M):
            with backpack(retain_graph=True):
                grads = autograd.grad(A[b, m], fc.weight, retain_graph=True)
            grad_autograd[b, m] = grads[0]

    grad_backpack = zeros(B, M, *fc.weight.shape)
    for i in range(M):
        with backpack(BatchGrad(), retain_graph=True):
            A[:, i].backward(ones_like(A[:, i]), retain_graph=True)
        grad_backpack[:, i] = fc.weight.grad_batch

    check_sizes_and_values(grad_backpack, grad_autograd)
