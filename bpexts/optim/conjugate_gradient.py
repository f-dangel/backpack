""" Conjugate gradient solver. Solve A*x = b with PSD matrix A"""

import torch


def cg(A, b, x0=None, tol=1E-5, maxiter=None, atol=None):
    """PyTorch implementation of conjugate gradient algorithm. Solve A*x=b.

    The interface is similar to CG provided by scipy.sparse.linalg.cg.
    Device is inferred by vector b.

    The main iteration loop follows the pseudo code from Wikipedia:
        https://en.wikipedia.org/w/index.php?title=Conjugate_gradient
        _method&oldid=855450922

    TODO: It is possible for the current implementation to get stuck
          if CG does not converge, since the maxiters cannot (yet) be
          infered from A if A is an implicit Hessian-vector product routine.

    Parameters:
    -----------
    A (torch.Tensor or function): Positive semi-definite matrix defining
                                  the linear system or a function providing
                                  matrix-vector products
    b (torch.Tensor): Tensor of shape (N,) denoting the right hand side
    x0 (torch.Tensor): Tensor of shape (N,), initial state
    tol (float): Maximum desired ratio of residuum norm and norm of b
    maxiter (int): Maximum number of iterations
    atol (float): Maximum value of residuum norm

    Returns:
    --------
    x (torch.Tensor): converged solution
    info (int): Provides convergence information
    """
    if not len(b.size()) == 1:
        raise ValueError('Expect 1d shape for b, got {}'.format(b.size()))
    device = b.device

    # matrix multiplication
    A_matmul = A.matmul if isinstance(A, torch.Tensor) else A

    # set initial value and convergence criterion
    atol = torch.tensor([0. if atol is None else atol], device=device)
    x = torch.zeros_like(b) if x0 is None else x0

    # initialize parameters
    r = b - A_matmul(x)
    p = r.clone()
    rs_old = r.matmul(r)

    # stopping criterion
    norm_bound = torch.tensor([tol * torch.norm(b), atol],
                              device=device,
                              dtype=rs_old.dtype)
    norm_bound = torch.max(norm_bound)

    def stopping_criterion(rs, numiter):
        """Check whether CG stops (convergence or steps exceeded)."""
        norm_converged = bool(torch.gt(norm_bound, torch.sqrt(rs)).item())
        info = 0 if norm_converged else numiter
        iters_exceeded = False if maxiter is None else (numiter > maxiter)
        return (norm_converged or iters_exceeded), info

    # iterate
    iterations = 0
    while True:
        Ap = A_matmul(p)
        alpha = rs_old / p.matmul(Ap)
        x.add_(alpha * p)
        r.sub_(alpha * Ap)
        rs_new = r.matmul(r)
        iterations += 1

        stop, info = stopping_criterion(rs_new, iterations)
        if stop:
            return x, info

        p.mul_(rs_new / rs_old)
        p.add_(r)
        rs_old = rs_new
