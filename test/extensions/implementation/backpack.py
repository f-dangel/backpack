"""Extension implementations with BackPACK."""
from test.extensions.implementation.base import ExtensionsImplementation
from test.extensions.implementation.hooks import (
    BatchL2GradHook,
    ExtensionHookManager,
    SumGradSquaredHook,
)
from test.extensions.problem import ExtensionsTestProblem
from test.utils import chunk_sizes
from typing import List

from torch import Tensor, cat, einsum

import backpack.extensions as new_ext
from backpack import backpack


class BackpackExtensions(ExtensionsImplementation):
    """Extension implementations with BackPACK."""

    def __init__(self, problem: ExtensionsTestProblem):
        """Add BackPACK functionality to, and store, the test case.

        Args:
            problem: Test case
        """
        problem.extend()
        super().__init__(problem)

    def batch_grad(self, subsampling) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.BatchGrad(subsampling=subsampling)):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
        return self.problem.collect_data("grad_batch")

    def batch_l2_grad(self) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.BatchL2Grad()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
        return self.problem.collect_data("batch_l2")

    def batch_l2_grad_extension_hook(self) -> List[Tensor]:
        """Individual gradient squared ℓ₂ norms via extension hook.

        Returns:
            Parameter-wise individual gradient norms.
        """
        hook = ExtensionHookManager(BatchL2GradHook())
        with backpack(new_ext.BatchGrad(), extension_hook=hook):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
        return self.problem.collect_data("batch_l2_hook")

    def sgs(self) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.SumGradSquared()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
        return self.problem.collect_data("sum_grad_squared")

    def sgs_extension_hook(self) -> List[Tensor]:
        """Individual gradient second moment via extension hook.

        Returns:
            Parameter-wise individual gradient second moment.
        """
        hook = ExtensionHookManager(SumGradSquaredHook())
        with backpack(new_ext.BatchGrad(), extension_hook=hook):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
        return self.problem.collect_data("sum_grad_squared_hook")

    def variance(self) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.Variance()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
        return self.problem.collect_data("variance")

    def diag_ggn(self) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.DiagGGNExact()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
        return self.problem.collect_data("diag_ggn_exact")

    def diag_ggn_exact_batch(self) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.BatchDiagGGNExact()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
        return self.problem.collect_data("diag_ggn_exact_batch")

    def diag_ggn_mc(self, mc_samples) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.DiagGGNMC(mc_samples=mc_samples)):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
        return self.problem.collect_data("diag_ggn_mc")

    def diag_ggn_mc_batch(self, mc_samples) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.BatchDiagGGNMC(mc_samples=mc_samples)):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
        return self.problem.collect_data("diag_ggn_mc_batch")

    def diag_ggn_mc_chunk(self, mc_samples: int, chunks: int = 10) -> List[Tensor]:
        """Like ``diag_ggn_mc``, but can handle more samples by chunking.

        Args:
            mc_samples: Number of Monte-Carlo samples.
            chunks: Maximum sequential split of the computation. Default: ``10``.

        Returns:
            Parameter-wise MC-approximation of the GGN diagonal.
        """
        chunk_samples = chunk_sizes(mc_samples, chunks)
        chunk_weights = [samples / mc_samples for samples in chunk_samples]

        diag_ggn_mc = None

        for weight, samples in zip(chunk_weights, chunk_samples):
            chunk_diag_ggn_mc = self.diag_ggn_mc(samples)
            chunk_diag_ggn_mc = [diag_mc * weight for diag_mc in chunk_diag_ggn_mc]

            if diag_ggn_mc is None:
                diag_ggn_mc = chunk_diag_ggn_mc
            else:
                for idx in range(len(diag_ggn_mc)):
                    diag_ggn_mc[idx] += chunk_diag_ggn_mc[idx]

        return diag_ggn_mc

    def diag_ggn_mc_batch_chunk(
        self, mc_samples: int, chunks: int = 10
    ) -> List[Tensor]:
        """Like ``diag_ggn_mc_batch``, but can handle more samples by chunking.

        Args:
            mc_samples: Number of Monte-Carlo samples.
            chunks: Maximum sequential split of the computation. Default: ``10``.

        Returns:
            Parameter-wise MC-approximation of the per-sample GGN diagonals.
        """
        chunk_samples = chunk_sizes(mc_samples, chunks)
        chunk_weights = [samples / mc_samples for samples in chunk_samples]

        diag_ggn_mc_batch = None

        for weight, samples in zip(chunk_weights, chunk_samples):
            chunk_diag_ggn_mc_batch = self.diag_ggn_mc_batch(samples)
            chunk_diag_ggn_mc_batch = [
                diag_mc * weight for diag_mc in chunk_diag_ggn_mc_batch
            ]

            if diag_ggn_mc_batch is None:
                diag_ggn_mc_batch = chunk_diag_ggn_mc_batch
            else:
                for idx in range(len(diag_ggn_mc_batch)):
                    diag_ggn_mc_batch[idx] += chunk_diag_ggn_mc_batch[idx]

        return diag_ggn_mc_batch

    def diag_h(self) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.DiagHessian()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
        return self.problem.collect_data("diag_h")

    def kfac(self, mc_samples: int = 1) -> List[List[Tensor]]:  # noqa:D102
        with backpack(new_ext.KFAC(mc_samples=mc_samples)):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
        return self.problem.collect_data("kfac")

    def kflr(self) -> List[List[Tensor]]:  # noqa:D102
        with backpack(new_ext.KFLR()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
        return self.problem.collect_data("kflr")

    def kfra(self) -> List[List[Tensor]]:  # noqa:D102
        with backpack(new_ext.KFRA()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
        return self.problem.collect_data("kfra")

    def diag_h_batch(self) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.BatchDiagHessian()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
        return self.problem.collect_data("diag_h_batch")

    def ggn(self, subsampling: List[int] = None) -> Tensor:  # noqa:D102
        return self._square_sqrt_ggn(self.sqrt_ggn(subsampling=subsampling))

    def sqrt_ggn(self, subsampling: List[int] = None) -> List[Tensor]:
        """Compute the matrix square root of the exact generalized Gauss-Newton.

        Args:
            subsampling: Indices of active samples. Defaults to ``None`` (use all
                samples in the mini-batch).

        Returns:
            Parameter-wise matrix square root of the exact GGN.
        """
        with backpack(new_ext.SqrtGGNExact(subsampling=subsampling)):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
        return self.problem.collect_data("sqrt_ggn_exact")

    def sqrt_ggn_mc(
        self, mc_samples: int, subsampling: List[int] = None
    ) -> List[Tensor]:
        """Compute the approximate matrix square root of the generalized Gauss-Newton.

        Args:
            mc_samples: Number of Monte-Carlo samples.
            subsampling: Indices of active samples. Defaults to ``None`` (use all
                samples in the mini-batch).

        Returns:
            Parameter-wise approximate matrix square root of the exact GGN.
        """
        with backpack(
            new_ext.SqrtGGNMC(mc_samples=mc_samples, subsampling=subsampling)
        ):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
        return self.problem.collect_data("sqrt_ggn_mc")

    def ggn_mc(
        self, mc_samples: int, chunks: int = 1, subsampling: List[int] = None
    ) -> Tensor:  # noqa:D102
        samples = chunk_sizes(mc_samples, chunks)
        weights = [samples / mc_samples for samples in samples]

        return sum(
            w * self._square_sqrt_ggn(self.sqrt_ggn_mc(s, subsampling=subsampling))
            for w, s in zip(weights, samples)
        )

    @staticmethod
    def _square_sqrt_ggn(sqrt_ggn: List[Tensor]) -> Tensor:
        """Utility function to concatenate and square the GGN factorization.

        Args:
            sqrt_ggn: Parameter-wise matrix square root of the GGN.

        Returns:
            Matrix representation of the GGN.
        """
        sqrt_mat = cat([s.flatten(start_dim=2) for s in sqrt_ggn], dim=2)
        return einsum("cni,cnj->ij", sqrt_mat, sqrt_mat)

    def kfac_chunk(self, mc_samples: int, chunks: int = 10) -> List[Tensor]:
        """Like ``kfac``, but can handle more samples by chunking.

        Args:
            mc_samples: Number of Monte-Carlo samples.
            chunks: Maximum sequential split of the computation. Default: ``10``.

        Returns:
            KFAC approximation from MC-sampling calculated via chunks.
        """
        chunk_samples = chunk_sizes(mc_samples, chunks)
        chunk_weights = [samples / mc_samples for samples in chunk_samples]

        kfac_mc = None

        for weight, samples in zip(chunk_weights, chunk_samples):
            chunk_kfac = self.kfac(samples)
            chunk_kfac = [[k * weight for k in kfac] for kfac in chunk_kfac]

            if kfac_mc is None:
                kfac_mc = chunk_kfac
            else:
                kfac_mc = [
                    [new_k + k for new_k, k in zip(new_kfac, kfac)]
                    for new_kfac, kfac in zip(chunk_kfac, kfac_mc)
                ]
        return kfac_mc
