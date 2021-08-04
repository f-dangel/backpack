"""Base class containing the functions to compare BackPACK and autograd."""
from abc import ABC, abstractmethod
from test.extensions.problem import ExtensionsTestProblem
from typing import List, Union

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
    def batch_grad(self, subsampling: Union[List[int], None]) -> List[Tensor]:
        """Individual gradients.

        Args:
            subsampling: List of active samples. ``None`` means all samples.
        """
        return

    @abstractmethod
    def batch_l2_grad(self) -> List[Tensor]:
        """L2 norm of Individual gradients."""
        return

    @abstractmethod
    def sgs(self) -> List[Tensor]:
        """Sum of Square of Individual gradients."""
        return

    @abstractmethod
    def variance(self) -> List[Tensor]:
        """Variance of Individual gradients."""
        return

    @abstractmethod
    def diag_ggn(self) -> List[Tensor]:
        """Diagonal of Gauss Newton."""
        return

    @abstractmethod
    def diag_ggn_exact_batch(self) -> List[Tensor]:
        """Individual diagonal of Generalized Gauss-Newton/Fisher."""
        return

    @abstractmethod
    def diag_ggn_mc(self, mc_samples: int) -> List[Tensor]:
        """MC approximation of the generalized Gauss-Newton/Fisher diagonal.

        Args:
            mc_samples: Number of Monte-Carlo samples used for the approximation.
        """
        return

    @abstractmethod
    def diag_ggn_mc_batch(self, mc_samples: int) -> List[Tensor]:
        """MC approximation of individual Generalized Gauss-Newton/Fisher diagonal.

        Args:
            mc_samples: Number of Monte-Carlo samples used for the approximation.
        """
        return

    @abstractmethod
    def diag_h(self) -> List[Tensor]:
        """Diagonal of Hessian."""
        return

    @abstractmethod
    def kfac(self, mc_samples: int = 1) -> List[List[Tensor]]:
        """Kronecker-factored approximate curvature (KFAC).

        Args:
            mc_samples: Number of Monte-Carlo samples. Default: ``1``.

        Returns:
            Parameter-wise lists of Kronecker factors.
        """
        return

    @abstractmethod
    def kflr(self) -> List[List[Tensor]]:
        """Kronecker-factored low-rank approximation (KFLR).

        Returns:
            Parameter-wise lists of Kronecker factors.
        """
        return

    @abstractmethod
    def kfra(self) -> List[List[Tensor]]:
        """Kronecker-factored recursive approximation (KFRA).

        Returns:
            Parameter-wise lists of Kronecker factors.
        """
        return

    @abstractmethod
    def diag_h_batch(self) -> List[Tensor]:
        """Per-sample Hessian diagonal.

        Returns:
            Parameter-wise per-sample Hessian diagonal.
        """
        return

    @abstractmethod
    def ggn(self, subsampling: List[int] = None) -> Tensor:
        """Exact generalized Gauss-Newton/Fisher matrix.

        Note:
            For losses with ``'mean'`` reduction, the GGN is ``¹/N ∑ₙ Jₙᵀ Hₙ Jₙ``. If
            sub-sampling is enabled, the sum will only run over active samples. The
            normalization will not be ``1/len(subsampling)``, but remain ``1/N``.

        Args:
            subsampling: Indices of active samples. Default: ``None`` (all).

        Returns:
            Matrix representation of the exact GGN.
        """
        return

    @abstractmethod
    def ggn_mc(
        self, mc_samples: int, chunks: int = 1, subsampling: List[int] = None
    ) -> Tensor:
        """Compute the MC-approximation of the GGN in chunks of MC samples.

        Args:
            mc_samples: Number of Monte-Carlo samples.
            chunks: Number of sequential portions to split the computation.
                Default: ``1`` (no sequential split).
            subsampling: Indices of active samples. Default: ``None`` (all).

        Returns:
            Matrix representation of the Monte-Carlo approximated GGN.
        """
        return
