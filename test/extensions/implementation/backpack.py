from test.extensions.implementation.base import ExtensionsImplementation
from test.extensions.implementation.hooks import (
    BatchL2GradHook,
    ExtensionHookManager,
    SumGradSquaredHook,
)

import backpack.extensions as new_ext
from backpack import backpack


class BackpackExtensions(ExtensionsImplementation):
    """Extension implementations with BackPACK."""

    def __init__(self, problem):
        problem.extend()
        super().__init__(problem)

    def batch_grad(self):
        with backpack(new_ext.BatchGrad()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            batch_grads = [p.grad_batch for p in self.problem.model.parameters()]
        return batch_grads

    def batch_l2_grad(self):
        with backpack(new_ext.BatchL2Grad()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            batch_l2_grad = [p.batch_l2 for p in self.problem.model.parameters()]
        return batch_l2_grad

    def batch_l2_grad_extension_hook(self):
        """Individual gradient squared ℓ₂ norms via extension hook."""
        hook = ExtensionHookManager(BatchL2GradHook())
        with backpack(new_ext.BatchGrad(), extension_hook=hook):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            batch_l2_grad = [p.batch_l2_hook for p in self.problem.model.parameters()]
        return batch_l2_grad

    def sgs(self):
        with backpack(new_ext.SumGradSquared()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            sgs = [p.sum_grad_squared for p in self.problem.model.parameters()]
        return sgs

    def sgs_extension_hook(self):
        """Individual gradient second moment via extension hook."""
        hook = ExtensionHookManager(SumGradSquaredHook())
        with backpack(new_ext.BatchGrad(), extension_hook=hook):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            sgs = [p.sum_grad_squared_hook for p in self.problem.model.parameters()]
        return sgs

    def variance(self):
        with backpack(new_ext.Variance()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            variances = [p.variance for p in self.problem.model.parameters()]
        return variances

    def diag_ggn(self):
        with backpack(new_ext.DiagGGNExact()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            diag_ggn = [p.diag_ggn_exact for p in self.problem.model.parameters()]
        return diag_ggn

    def diag_ggn_exact_batch(self):
        with backpack(new_ext.BatchDiagGGNExact()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            diag_ggn_exact_batch = [
                p.diag_ggn_exact_batch for p in self.problem.model.parameters()
            ]
        return diag_ggn_exact_batch

    def diag_ggn_mc(self, mc_samples):
        with backpack(new_ext.DiagGGNMC(mc_samples=mc_samples)):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            diag_ggn_mc = [p.diag_ggn_mc for p in self.problem.model.parameters()]
        return diag_ggn_mc

    def diag_ggn_mc_batch(self, mc_samples):
        with backpack(new_ext.BatchDiagGGNMC(mc_samples=mc_samples)):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            diag_ggn_mc_batch = [
                p.diag_ggn_mc_batch for p in self.problem.model.parameters()
            ]
        return diag_ggn_mc_batch

    def diag_ggn_mc_chunk(self, mc_samples, chunks=10):
        """Like ``diag_ggn_mc``, but handles larger number of samples by chunking."""
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

    def diag_ggn_mc_batch_chunk(self, mc_samples, chunks=10):
        """
        Like ``diag_ggn_mc_batch``, but handles larger number of samples by chunking.
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
    def chunk_sizes(total_size, num_chunks):
        """Return list containing the sizes of chunks."""
        chunk_size = max(total_size // num_chunks, 1)

        if chunk_size == 1:
            sizes = total_size * [chunk_size]
        else:
            equal, rest = divmod(total_size, chunk_size)
            sizes = equal * [chunk_size]

            if rest != 0:
                sizes.append(rest)

        return sizes

    def diag_h(self):
        with backpack(new_ext.DiagHessian()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            diag_h = [p.diag_h for p in self.problem.model.parameters()]
        return diag_h

    def kfac(self, mc_samples=1):
        with backpack(new_ext.KFAC(mc_samples=mc_samples)):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            kfac = [p.kfac for p in self.problem.model.parameters()]

        return kfac

    def kflr(self):
        with backpack(new_ext.KFLR()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            kflr = [p.kflr for p in self.problem.model.parameters()]

        return kflr

    def kfra(self):
        with backpack(new_ext.KFRA()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            kfra = [p.kfra for p in self.problem.model.parameters()]

        return kfra

    def diag_h_batch(self):
        with backpack(new_ext.BatchDiagHessian()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            diag_h_batch = [p.diag_h_batch for p in self.problem.model.parameters()]

        return diag_h_batch
