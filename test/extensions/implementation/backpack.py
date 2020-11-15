from test.extensions.implementation.base import ExtensionsImplementation

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

    def sgs(self):
        with backpack(new_ext.SumGradSquared()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            sgs = [p.sum_grad_squared for p in self.problem.model.parameters()]
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

    def diag_ggn_mc(self, mc_samples):
        with backpack(new_ext.DiagGGNMC(mc_samples=mc_samples)):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            diag_ggn_mc = [p.diag_ggn_mc for p in self.problem.model.parameters()]
        return diag_ggn_mc

    def diag_h(self):
        with backpack(new_ext.DiagHessian()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            diag_h = [p.diag_h for p in self.problem.model.parameters()]
        return diag_h
