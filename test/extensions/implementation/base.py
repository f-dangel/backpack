"""Base class containing the functions to compare BackPACK and autograd."""
from typing import List

from torch import Tensor


class ExtensionsImplementation:
    """Base class for autograd and BackPACK implementations of extensions."""

    def __init__(self, problem):
        self.problem = problem

    def batch_grad(self):
        """Individual gradients."""
        raise NotImplementedError

    def batch_l2_grad(self):
        """L2 norm of Individual gradients."""
        raise NotImplementedError

    def sgs(self):
        """Sum of Square of Individual gradients"""
        raise NotImplementedError

    def variance(self):
        """Variance of Individual gradients"""
        raise NotImplementedError

    def diag_ggn(self):
        """Diagonal of Gauss Newton"""
        raise NotImplementedError

    def diag_ggn_batch(self):
        """Individual diagonal of Generalized Gauss-Newton/Fisher"""
        raise NotImplementedError

    def diag_ggn_mc(self, mc_samples):
        """MC approximation of Diagonal of Gauss Newton"""
        raise NotImplementedError

    def diag_ggn_mc_batch(self, mc_samples):
        """MC approximation of individual Generalized Gauss-Newton/Fisher diagonal."""
        raise NotImplementedError

    def diag_h(self):
        """Diagonal of Hessian"""
        raise NotImplementedError

    def kfac(self, mc_samples: int = 1) -> List[List[Tensor]]:
        """Kronecker-factored approximate curvature (KFAC).

        Args:
            mc_samples: Number of Monte-Carlo samples. Default: ``1``.

        Returns:
            Parameter-wise lists of Kronecker factors.
        """
        raise NotImplementedError

    def kflr(self) -> List[List[Tensor]]:
        """Kronecker-factored low-rank approximation (KFLR).

        Returns:
            Parameter-wise lists of Kronecker factors.
        """
        raise NotImplementedError

    def kfra(self) -> List[List[Tensor]]:
        """Kronecker-factored recursive approximation (KFRA).

        Returns:
            Parameter-wise lists of Kronecker factors.
        """

    def diag_h_batch(self) -> List[Tensor]:
        """Per-sample Hessian diagonal.

        Returns:
            list(torch.Tensor): Parameter-wise per-sample Hessian diagonal.
        """
        raise NotImplementedError

    def ggn(self) -> Tensor:
        """Exact generalized Gauss-Newton/Fisher matrix.

        Returns:
            Matrix representation of the exact GGN.
        """
        raise NotImplementedError
