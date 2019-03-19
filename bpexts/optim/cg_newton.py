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
        r"""Solve :math:`Ax = b` for :math:`x` using conjugate gradient.

        Parameters
        ----------
        A : torch.Tensor or function
            Matrix of the linear operator `A` or function implementing
            matrix-vector multiplication by `A`.
        b : torch.Tensor
            Right-hand side of the linear system
        cg_atol: float
            Absolute tolerance to accept convergence. Stop if
            :math:`|| A x - b || <` `cg_atol`
        cg_tol: float
            Relative tolerance to accept convergence. Stop if
            :math:`|| A x - b || / || b || <` `cg_atol`.
        cg_maxiter: int
            Maximum number of iterations

        Note
        ----
        `A` has to provide a `matmul` function.

        Returns
        -------
        torch.Tensor
            Approximate solution :math:`x` of the linear system
        """
        x, info = cg(A, b, tol=cg_tol, atol=cg_atol, maxiter=cg_maxiter)
        if info != 0:
            warn('CG WARNING: No convergence: {}'.format(info))
        return x


class CGNewton(CGSolver):
    r"""Solve for update :math:`d` using CG to solve
    :math:`[(1 - \alpha)H + \alpha I] d = -g`.

    This update rule is inspired by the work of Chen et al.: BDA-PCH
    (2018)
    
    Note
    ----
    Usually, :math:`H` is an approximation to the curvature matrix
    (Hessian, Fisher, Generalized Gauss-Newton) and :math:`g` is the
    gradient of the objective function with respect to the parameters.
    """
    def __init__(self, params, lr, alpha, cg_atol=1E-8, cg_tol=1E-5,
                 cg_maxiter=None):
        r"""Compute Newton updates, use average of identity and Hessian.

        Solves

        .. math::

            [(1 - \alpha) H + alpha * I] * d = -g

        Applies
        
        .. math::

            \theta \leftarrow \theta + \gamma d

        with :math:`\gamma` given by `lr`.

        Parameters
        ----------
        lr : float
            Learning rate :math:`\gamma`
        alpha :  float between 0 and 1
            Ratio :math:`\alpha` for average of Hessian and identity matrix
        cg_atol : float, optional
            Absolute tolerance for CG convergence
        cg_tol : float, optional
            Relative tolerance for CG convergence
        cg_maxiter :  int, optional
            Maximum number of iterations. Per default, the number of
            iterations is unlimited
        """
        defaults = dict(lr=lr, alpha=alpha, cg_atol=cg_atol, cg_tol=cg_tol,
                        cg_maxiter=cg_maxiter)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Perform a single optimization step.
        
        Parameters
        ----------
        closure : function, optional
            See PyTorch documentation
        """
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
                warn('User Warning: Encountered parameter with None'
                     ' grad attribute of size {}'.format(p.size()))
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
        r"""Regularized Hessian-vector product with identity.

        Parameters
        ----------
        mvp : function
            Implicit matrix-vector multiplication routine
        alpha : float, between 0 and 1, optional
            Averaging constant for the identity. Per default,
            no additional regularization is applied

        Returns
        -------
        function
            Implicit matrix-vector multiplication with the
            provided routine scaled by :math:`(1 - \alpha)` plus
            multiplication by :math:`\alpha I` where :math:`I` denotes
            the identity matrix

            .. math::

                (1 - \alpha) \mathrm{mvp}(v) + \alpha v.
        """
        def mvp_avg_id(v):
            r"""Multiply vector by weighted average of identity and `mvp`.

            Parameters
            ----------
            v : torch.Tensor
                One-dimensional tensor representing a vector

            Returns
            -------
            mvp_avg_id : function
                Implicit matrix vector multiplication as follows:

                .. math::

                    (1 - \alpha) mvp(v) + \alpha v.
            """
            return self.matmul_average_identity(mvp, v, alpha=alpha)
        return mvp_avg_id

    @staticmethod
    def matmul_average_identity(mvp, v, alpha=0):
        r"""Perform matrix multiplication by (1 - alpha) * mvp + alpha * Id.

        Use weighted average between matrix and identity.

        Parameters
        ----------
        mvp : (function)
            Function mapping v to matrix * v
        v : (torch.Tensor)
            Vector being multiplied
        alpha : (float between 0 and 1)
            Average ratio

        Returns
        -------
        mv : (torch.Tensor)
            Result of [(1 - alpha) linop + alpha * Id] * v
        """
        if not 0 <= alpha <= 1:
            raise ValueError('Invalid alpha: {} not in [0; 1]'.format(alpha))
        return alpha * v + (1 - alpha) * mvp(v)
