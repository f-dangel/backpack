"""Base class containing the functions to compare BackPACK and autograd."""
from abc import ABC, abstractmethod
from test.extensions.problem import ExtensionsTestProblem
from typing import List

from torch import Tensor


class ExtensionsImplementation(ABC):
    """Base class for autograd and BackPACK implementations of extensions."""

    def __init__(self, problem: ExtensionsTestProblem):
        """Store the test case.

        Args:
            problem: Test case.
        """
        self.problem = problem

    @abstractmethod
    def batch_grad(self) -> List[Tensor]:
        """Individual gradients."""
        raise NotImplementedError

    @abstractmethod
    def batch_l2_grad(self) -> List[Tensor]:
        """L2 norm of Individual gradients."""
        raise NotImplementedError

    @abstractmethod
    def sgs(self) -> List[Tensor]:
        """Sum of Square of Individual gradients."""
        raise NotImplementedError

    @abstractmethod
    def variance(self) -> List[Tensor]:
        """Variance of Individual gradients."""
        raise NotImplementedError

    @abstractmethod
    def diag_ggn(self) -> List[Tensor]:
        """Diagonal of Gauss Newton."""
        raise NotImplementedError

    @abstractmethod
    def diag_ggn_exact_batch(self) -> List[Tensor]:
        """Individual diagonal of Generalized Gauss-Newton/Fisher."""
        raise NotImplementedError

    @abstractmethod
    def diag_ggn_mc(self, mc_samples: int) -> List[Tensor]:
        """MC approximation of the generalized Gauss-Newton/Fisher diagonal.

        Args:
            mc_samples: Number of Monte-Carlo samples used for the approximation.
        """
        raise NotImplementedError

    @abstractmethod
    def diag_ggn_mc_batch(self, mc_samples: int) -> List[Tensor]:
        """MC approximation of individual Generalized Gauss-Newton/Fisher diagonal.

        Args:
            mc_samples: Number of Monte-Carlo samples used for the approximation.
        """
        raise NotImplementedError

    @abstractmethod
    def kfac(self, mc_samples: int = 1) -> List[List[Tensor]]:
        """Kronecker-factored approximate curvature (KFAC).

        Args:
            mc_samples: Number of Monte-Carlo samples. Default: ``1``.

        Returns:
            Parameter-wise lists of Kronecker factors.
        """
        raise NotImplementedError

    @abstractmethod
    def kflr(self) -> List[List[Tensor]]:
        """Kronecker-factored low-rank approximation (KFLR).

        Returns:
            Parameter-wise lists of Kronecker factors.
        """
        raise NotImplementedError

    @abstractmethod
    def kfra(self) -> List[List[Tensor]]:
        """Kronecker-factored recursive approximation (KFRA).

        Returns:
            Parameter-wise lists of Kronecker factors.
        """
        raise NotImplementedError

    @abstractmethod
    def diag_h(self) -> List[Tensor]:
        """Diagonal of Hessian.

        Returns:
            Hessian diagonal for each parameter.
        """
        raise NotImplementedError

    @abstractmethod
    def diag_h_batch(self) -> List[Tensor]:
        """Per-sample Hessian diagonal.

        Returns:
            list(torch.Tensor): Parameter-wise per-sample Hessian diagonal.
        """
        raise NotImplementedError

    @abstractmethod
    def ggn(self) -> Tensor:
        """Exact generalized Gauss-Newton/Fisher matrix.

        Returns:
            Matrix representation of the exact GGN.
        """
        raise NotImplementedError

    @abstractmethod
    def ggn_mc(self, mc_samples: int, chunks: int = 1) -> Tensor:
        """Compute the MC-approximation of the GGN in chunks of MC samples.

        Args:
            mc_samples: Number of Monte-Carlo samples.
            chunks: Number of sequential portions to split the computation.
                Default: ``1`` (no sequential split).

        Returns:
            Matrix representation of the Monte-Carlo approximated GGN.
        """
        raise NotImplementedError
