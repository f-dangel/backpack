import torch

import backpack.extensions as new_ext
from backpack import backpack
from backpack.extensions.curvature import Curvature
from backpack.extensions.secondorder.hbp import (
    BackpropStrategy,
    ExpectationApproximation,
    LossHessianStrategy,
)
from backpack.utils.kroneckers import kfacs_to_mat

from .implementation import Implementation


class BpextImpl(Implementation):
    def gradient(self):
        return list(torch.autograd.grad(self.loss(), self.model.parameters()))

    def batch_gradients(self):
        with backpack(new_ext.BatchGrad()):
            self.loss().backward()
            batch_grads = [p.grad_batch for p in self.model.parameters()]
        return batch_grads

    def batch_l2(self):
        with backpack(new_ext.BatchL2Grad()):
            self.loss().backward()
            batch_l2s = [p.batch_l2 for p in self.model.parameters()]
        return batch_l2s

    def variance(self):
        with backpack(new_ext.Variance()):
            self.loss().backward()
            variances = [p.variance for p in self.model.parameters()]
        return variances

    def sgs(self):
        with backpack(new_ext.SumGradSquared()):
            self.loss().backward()
            sgs = [p.sum_grad_squared for p in self.model.parameters()]
        return sgs

    def diag_ggn(self):
        with backpack(new_ext.DiagGGN()):
            self.loss().backward()
            diag_ggn = [p.diag_ggn for p in self.model.parameters()]
        return diag_ggn

    def diag_ggn_mc(self):
        with backpack(new_ext.DiagGGNMC()):
            self.loss().backward()
            diag_ggn = [p.diag_ggn_mc for p in self.model.parameters()]
        return diag_ggn

    def diag_h(self):
        with backpack(new_ext.DiagHessian()):
            self.loss().backward()
            diag_h = [p.diag_h for p in self.model.parameters()]
        return diag_h

    def hvp(self, vec_list):
        return self.hmp(vec_list)

    def hmp(self, mat_list):
        assert len(mat_list) == len(list(self.model.parameters()))
        results = []
        with backpack(new_ext.HMP()):
            self.loss().backward()
            for p, mat in zip(self.model.parameters(), mat_list):
                results.append(p.hmp(mat))
        return results

    def ggn_mp(self, mat_list):
        assert len(mat_list) == len(list(self.model.parameters()))
        results = []
        with backpack(new_ext.GGNMP()):
            self.loss().backward()
            for p, mat in zip(self.model.parameters(), mat_list):
                results.append(p.ggnmp(mat))
        return results

    def ggn_vp(self, vec_list):
        return self.ggn_mp(vec_list)

    def pchmp(self, mat_list, modify):
        assert len(mat_list) == len(list(self.model.parameters()))
        results = []
        with backpack(new_ext.PCHMP(modify=modify)):
            self.loss().backward()
            for p, mat in zip(self.model.parameters(), mat_list):
                results.append(p.pchmp(mat))
        return results

    def matrices_from_kronecker_curvature(self, extension_cls, savefield):
        results = []
        with backpack(extension_cls()):
            self.loss().backward()
            for p in self.model.parameters():
                factors = getattr(p, savefield)
                results.append(kfacs_to_mat(factors))
        return results

    def kfra_blocks(self):
        return self.matrices_from_kronecker_curvature(new_ext.KFRA, "kfra")

    def kflr_blocks(self):
        return self.matrices_from_kronecker_curvature(new_ext.KFLR, "kflr")

    def kfac_blocks(self):
        return self.matrices_from_kronecker_curvature(new_ext.KFAC, "kfac")

    def hbp_with_curv(
        self,
        curv_type,
        loss_hessian_strategy=LossHessianStrategy.SUM,
        backprop_strategy=BackpropStrategy.BATCH_AVERAGE,
        ea_strategy=ExpectationApproximation.BOTEV_MARTENS,
    ):
        results = []
        with backpack(
            new_ext.HBP(
                curv_type=curv_type,
                loss_hessian_strategy=loss_hessian_strategy,
                backprop_strategy=backprop_strategy,
                ea_strategy=ea_strategy,
            )
        ):
            self.loss().backward()
            for p in self.model.parameters():
                factors = p.hbp
                results.append(kfacs_to_mat(factors))
        return results

    def hbp_single_sample_ggn_blocks(self):
        return self.hbp_with_curv(Curvature.GGN)

    def hbp_single_sample_h_blocks(self):
        return self.hbp_with_curv(Curvature.HESSIAN)
