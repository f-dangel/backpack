"""Tests for conjugate gradient solver."""

import torch
from bpexts.optim.conjugate_gradient import cg


def simple(use_gpu=False):
    """Simple test scenario from Wikipedia on GPU/CPU.

    Parameters:
    -----------
    use_gpu (bool): Use GPU, else fall back to CPU
    """
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    # Wikipedia example
    A = torch.tensor([[4., 1.], [1., 3.]], device=device)
    b = torch.tensor([1., 2.], device=device)
    x0 = torch.tensor([2., 1.], device=device)

    # solve
    x, info = cg(A, b, x0=x0, maxiter=10)

    # check
    x_correct = torch.tensor([1. / 11., 7. / 11.], device=device)
    assert torch.allclose(x, x_correct)


def test_simple_on_cpu():
    """Run simple test on CPU."""
    simple(use_gpu=False)


def test_simple_on_gpu():
    """Try running simple test on GPU."""
    if torch.cuda.is_available():
        simple(use_gpu=True)


def random_task(dim, use_gpu=False):
    """Create a random linear system of fixed dimension.

    For better reproducibility, the seeds are set so zero.

    Parameters:
    -----------
    dim (int): Dimension of the system
    use_gpu (bool): Store quantities on GPU, else CPU

    Returns:
    --------
    A (torch.Tensor): Random regularized PSD matrix describing the system
    b (torch.Tensor): Right hand side of the linear system
    """
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    # random PSD matrix
    A = torch.rand(dim, dim, device=device)
    A = A.t().matmul(A) / torch.norm(A)**2
    b = torch.rand(dim, device=device)
    # regularize
    diag = torch.diagflat(torch.ones_like(b))
    A = A + diag
    return A, b


def random(dim=10, use_gpu=False):
    """Test CG for a random PSD matrix on GPU/CPU.

    Parameters:
    -----------
    dim (int): Dimension of the random linear system
    use_gpu (bool): Use GPU, else fall back to CPU
    """
    A, b = random_task(dim, use_gpu=use_gpu)
    x, info = cg(A, b, maxiter=dim)
    res = torch.norm(A.matmul(x) - b) / torch.norm(b)
    assert torch.allclose(res, torch.zeros_like(res), atol=1E-5)


def test_random_on_cpu():
    """Run random test on CPU."""
    random(use_gpu=False)


def test_random_on_gpu():
    """Try running random test on GPU."""
    if torch.cuda.is_available():
        random(use_gpu=True)


def implicit(dim=10, use_gpu=False):
    """Test CG with a function that provides matrix-vector products.

    Parameters:
    -----------
    dim (int): Dimension of the random linear system
    use_gpu (bool): Use GPU, else fall back to CPU
    """
    A, b = random_task(dim, use_gpu=use_gpu)

    def A_matmul(x):
        """Matrix multiplication by A.

        Parameters:
        -----------
        x (torch.Tensor): Vector of shape (N,)

        Returns:
        --------
        (torch.Tensor): Vector of shape (N,) representing A*x
        """
        return A.matmul(x)

    x, info = cg(A_matmul, b, maxiter=dim)
    res = torch.norm(A_matmul(x) - b) / torch.norm(b)
    assert torch.allclose(res, torch.zeros_like(res), atol=1E-5)


def test_implicit_on_cpu():
    """Run implicit test on CPU."""
    implicit(use_gpu=False)


def test_implicit_on_gpu():
    """Try running implicit test on GPU."""
    if torch.cuda.is_available():
        implicit(use_gpu=True)
