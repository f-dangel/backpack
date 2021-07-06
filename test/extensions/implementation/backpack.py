"""Extension implementations with BackPACK."""
from test.extensions.implementation.base import ExtensionsImplementation
from test.extensions.implementation.hooks import (
    BatchL2GradHook,
    ExtensionHookManager,
    SumGradSquaredHook,
)
from test.extensions.problem import ExtensionsTestProblem
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

    def batch_grad(self) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.BatchGrad()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            batch_grads = [p.grad_batch for p in self.problem.trainable_parameters()]
        return batch_grads

    def batch_l2_grad(self) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.BatchL2Grad()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            batch_l2_grad = [p.batch_l2 for p in self.problem.trainable_parameters()]
        return batch_l2_grad

    def batch_l2_grad_extension_hook(self) -> List[Tensor]:
        """Individual gradient squared ℓ₂ norms via extension hook.

        Returns:
            Parameter-wise individual gradient norms.
        """
        hook = ExtensionHookManager(BatchL2GradHook())
        with backpack(new_ext.BatchGrad(), extension_hook=hook):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            batch_l2_grad = [
                p.batch_l2_hook for p in self.problem.trainable_parameters()
            ]
        return batch_l2_grad

    def sgs(self) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.SumGradSquared()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            sgs = [p.sum_grad_squared for p in self.problem.trainable_parameters()]
        return sgs

    def sgs_extension_hook(self) -> List[Tensor]:
        """Individual gradient second moment via extension hook.

        Returns:
            Parameter-wise individual gradient second moment.
        """
        hook = ExtensionHookManager(SumGradSquaredHook())
        with backpack(new_ext.BatchGrad(), extension_hook=hook):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            sgs = [p.sum_grad_squared_hook for p in self.problem.trainable_parameters()]
        return sgs

    def variance(self) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.Variance()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            variances = [p.variance for p in self.problem.trainable_parameters()]
        return variances

    def diag_ggn(self) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.DiagGGNExact()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            diag_ggn = [p.diag_ggn_exact for p in self.problem.trainable_parameters()]
        return diag_ggn

    def diag_ggn_exact_batch(self) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.BatchDiagGGNExact()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            diag_ggn_exact_batch = [
                p.diag_ggn_exact_batch for p in self.problem.trainable_parameters()
            ]
        return diag_ggn_exact_batch

    def diag_ggn_mc(self, mc_samples) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.DiagGGNMC(mc_samples=mc_samples)):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            diag_ggn_mc = [p.diag_ggn_mc for p in self.problem.trainable_parameters()]
        return diag_ggn_mc

    def diag_ggn_mc_batch(self, mc_samples) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.BatchDiagGGNMC(mc_samples=mc_samples)):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            diag_ggn_mc_batch = [
                p.diag_ggn_mc_batch for p in self.problem.trainable_parameters()
            ]
        return diag_ggn_mc_batch

    def diag_ggn_mc_chunk(self, mc_samples: int, chunks: int = 10) -> List[Tensor]:
        """Like ``diag_ggn_mc``, but can handle more samples by chunking.

        Args:
            mc_samples: Number of Monte-Carlo samples.
            chunks: Maximum sequential split of the computation. Default: ``10``.

        Returns:
            Parameter-wise MC-approximation of the GGN diagonal.
        """
        chunk_samples = self.chunk_sizes(mc_samples, chunks)
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
        chunk_samples = self.chunk_sizes(mc_samples, chunks)
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

    @staticmethod
    def chunk_sizes(total_size: int, num_chunks: int) -> List[int]:
        """Return list containing the sizes of chunks.

        Args:
            total_size: Total computation work.
            num_chunks: Maximum number of chunks the work will be split into.

        Returns:
            List of chunks with split work.
        """
        chunk_size = max(total_size // num_chunks, 1)

        if chunk_size == 1:
            sizes = total_size * [chunk_size]
        else:
            equal, rest = divmod(total_size, chunk_size)
            sizes = equal * [chunk_size]

            if rest != 0:
                sizes.append(rest)

        return sizes

    def diag_h(self) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.DiagHessian()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            diag_h = [p.diag_h for p in self.problem.trainable_parameters()]
        return diag_h

    def kfac(self, mc_samples: int = 1) -> List[List[Tensor]]:  # noqa:D102
        with backpack(new_ext.KFAC(mc_samples=mc_samples)):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            kfac = [p.kfac for p in self.problem.trainable_parameters()]

        return kfac

    def kflr(self) -> List[List[Tensor]]:  # noqa:D102
        with backpack(new_ext.KFLR()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            kflr = [p.kflr for p in self.problem.trainable_parameters()]

        return kflr

    def kfra(self) -> List[List[Tensor]]:  # noqa:D102
        with backpack(new_ext.KFRA()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            kfra = [p.kfra for p in self.problem.trainable_parameters()]

        return kfra

    def diag_h_batch(self) -> List[Tensor]:  # noqa:D102
        with backpack(new_ext.BatchDiagHessian()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            diag_h_batch = [p.diag_h_batch for p in self.problem.trainable_parameters()]

        return diag_h_batch

    def ggn(self) -> Tensor:  # noqa:D102
        return self._square_sqrt_ggn(self.sqrt_ggn())

    def sqrt_ggn(self) -> List[Tensor]:
        """Compute the matrix square root of the exact generalized Gauss-Newton.

        Returns:
            Parameter-wise matrix square root of the exact GGN.
        """
        with backpack(new_ext.SqrtGGNExact()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()

        return [p.sqrt_ggn_exact for p in self.problem.trainable_parameters()]

    def sqrt_ggn_mc(self, mc_samples: int) -> List[Tensor]:
        """Compute the approximate matrix square root of the generalized Gauss-Newton.

        Args:
            mc_samples: Number of Monte-Carlo samples.

        Returns:
            Parameter-wise approximate matrix square root of the exact GGN.
        """
        with backpack(new_ext.SqrtGGNMC(mc_samples=mc_samples)):
            _, _, loss = self.problem.forward_pass()
            loss.backward()

        return [p.sqrt_ggn_mc for p in self.problem.trainable_parameters()]

    def ggn_mc(self, mc_samples: int, chunks: int = 1) -> Tensor:  # noqa:D102
        samples = self.chunk_sizes(mc_samples, chunks)
        weights = [samples / mc_samples for samples in samples]

        return sum(
            w * self._square_sqrt_ggn(self.sqrt_ggn_mc(s))
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
