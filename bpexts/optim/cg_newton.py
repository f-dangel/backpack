"""
Newton-style optimizer using implicit multiplication by the Hessian
to solve for updates by conjugate gradients.
"""
from warnings import warn
from torch.optim import Optimizer
from .conjugate_gradient import cg


class CGSolver(Optimizer):
    """PyTorch optimizer class extended by conjugate gradient method."""
    @staticmethod
    def solve(A, b, cg_atol, cg_tol, cg_maxiter):
        """Solve A*x = b for x using CG.

        Parameters:
        -----------
        A : (torch.Tensor or function)
            Matrix of the linear operator `A` or function implementing
            matrix-vector multiplication by `A`
        b : (torch.Tensor)
            Right-hand side of the linear system

        Returns:
        --------
        x :  (torch.Tensor)
            Approximate solution of the linear system
        """
        x, info = cg(A, b, tol=cg_tol, atol=cg_atol, maxiter=cg_maxiter)
        if info != 0:
            warn('CG WARNING: No convergence: {}'.format(info))
        return x


class CGNewton(CGSolver):
    """Solve for update d using CG to solve H * d = -g with gradient g."""

    def __init__(self, params, lr, alpha, cg_atol=1E-8, cg_tol=1E-5,
                 cg_maxiter=None):
        """Compute Newton updates by CG, use average of I and Hessian.

        Solves
            [(1 - alpha) * H + alpha * I] * d = -g
        Applies
            param -> param + alpha * d

        Parameters:
        -----------
        lr : (float)
            Learning rate
        alpha :  (float between 0 and 1)
            Ratio for average of Hessian and Id
        cg_atol, cg_tol : (float, float)
            Convergence parameters, see `scipy.sparse.linalg.cg`
        cg_maxiter :  (int)
            Maximum number of iterations
        """
        defaults = dict(lr=lr, alpha=alpha, cg_atol=cg_atol, cg_tol=cg_tol,
                        cg_maxiter=cg_maxiter)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            self.update_group(group)
        return loss

    def update_group(self, group):
        """Update parameter group according to optimizer strategy."""
        alpha = group['alpha']
        cg_atol = group['cg_atol']
        cg_tol = group['cg_tol']
        cg_maxiter = group['cg_maxiter']

        for p in group['params']:
            if p.grad is None:
                continue
            # prepare linear system
            g = p.grad.data
            b = -1 * g.view(-1)
            hvp_modified = self.hvp_for_update(p.hvp, alpha=alpha)
            # solve and update
            solution = self.solve(hvp_modified, b, cg_atol, cg_tol,
                                  cg_maxiter)
            solution = solution.view(p.size())
            p.data.add_(group['lr'], solution)

    def hvp_for_update(self, mvp, alpha=0):
        """Regularize Hessian-vector product with identity.

        Parameters:
        -----------
        mvp : (function)
            Implicit matrix-vector multiplication routine
        alpha : (float between 0 and 1)
            Averaging constant for the identity

        Returns:
        --------
        mvp_avg_id : (function)
            Implicit matrix-vector multiplication with the
            provided routine scaled (1 - alpha) plus multiplication
            by alpha * I (I is the identity)
        """
        def mvp_avg_id(v):
            """Multiply vector by weighted average of identity and hvp.

            Parameters:
            -----------
            v : (torch.Tensor)
                One-dimensional tensor representing a vector

            Returns:
            --------
            mvp_avg_id : (function)
                Function representing implicit matrix vector multiplication
                as follows: (1 - alpha) * mvp(v) + alpha * v.
            """
            return self.matmul_average_identity(mvp, v, alpha=alpha)
        return mvp_avg_id

    @staticmethod
    def matmul_average_identity(mvp, v, alpha=0):
        """Perform matrix multiplication by (1 - alpha) * mvp + alpha * Id.

        Use weighted average between matrix and identity.

        Parameters:
        -----------
        mvp : (function)
            Function mapping v to matrix * v
        v : (torch.Tensor)
            Vector being multiplied
        alpha : (float between 0 and 1)
            Average ratio

        Returns:
        --------
        mv : (torch.Tensor)
            Result of [(1 - alpha) linop + alpha * Id] * v
        """
        if not 0 <= alpha <= 1:
            raise ValueError('Invalid alpha: {} not in [0; 1]'.format(alpha))
        return alpha * v + (1 - alpha) * mvp(v)
