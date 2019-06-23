"""Test of conjugate gradient Newton style optimizer."""

import torch
from bpexts.optim.cg_newton import CGNewton


def simple(use_gpu=False):
    """Simple test scenario from Wikipedia on GPU/CPU.

    Parameters:
    -----------
    use_gpu (bool): Use GPU, else fall back to CPU
    """
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    # Wikipedia example: Minimize 0.5 * x^T * A * x - b^T * x
    A = torch.tensor([[4., 1.], [1., 3.]], device=device)
    b = torch.tensor([[1.], [2.]], device=device)
    x = torch.tensor([[2.], [1.]], device=device)
    x.requires_grad = True

    optimizer = CGNewton([x], lr=1., alpha=0., cg_maxiter=10)
    optimization_steps = 1
    for _ in range(optimization_steps):
        loss = 0.5 * x.t() @ A @ x - b.t() @ x
        optimizer.zero_grad()
        loss.backward()
        # Hessian-vector product
        x.hvp = A.matmul
        optimizer.step()
    x_correct = torch.tensor([[1. / 11.], [7. / 11.]], device=device)
    assert torch.allclose(x, x_correct, atol=1E-5)


def test_simple_on_cpu():
    """Run simple test on CPU."""
    simple(use_gpu=False)


def test_simple_on_gpu():
    """Run simple test on CPU."""
    if torch.cuda.is_available():
        simple(use_gpu=True)
