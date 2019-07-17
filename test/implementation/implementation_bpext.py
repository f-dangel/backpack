import torch
from .implementation import Implementation
from backpack import backpack
import backpack.extensions as ext
from backpack.curvature import Curvature
from backpack.secondorder.utils import matrix_from_kron_facs


class BpextImpl(Implementation):
    def gradient(self):
        return list(torch.autograd.grad(self.loss(), self.model.parameters()))

    def batch_gradients(self):
        with backpack(ext.BATCH_GRAD()):
            self.loss().backward()
            batch_grads = [p.grad_batch for p in self.model.parameters()]
        return batch_grads

    def batch_l2(self):
        with backpack(ext.BATCH_L2()):
            self.loss().backward()
            batch_l2s = [p.batch_l2 for p in self.model.parameters()]
        return batch_l2s

    def variance(self):
        with backpack(ext.VARIANCE()):
            self.loss().backward()
            variances = [p.variance for p in self.model.parameters()]
        return variances

    def sgs(self):
        with backpack(ext.SUM_GRAD_SQUARED()):
            self.loss().backward()
            sgs = [p.sum_grad_squared for p in self.model.parameters()]
        return sgs

    def diag_ggn(self):
        with backpack(ext.DIAG_GGN()):
            self.loss().backward()
            diag_ggn = [p.diag_ggn for p in self.model.parameters()]
        return diag_ggn

    def diag_h(self):
        with backpack(ext.DIAG_H()):
            self.loss().backward()
            diag_h = [p.diag_h for p in self.model.parameters()]
        return diag_h

    def hmp(self, mat_list):
        assert len(mat_list) == len(list(self.model.parameters()))
        results = []
        with backpack(ext.CMP(Curvature.HESSIAN)):
            self.loss().backward()
            for p, mat in zip(self.model.parameters(), mat_list):
                results.append(p.cmp(mat))
        return results

    def ggn_mp(self, mat_list):
        assert len(mat_list) == len(list(self.model.parameters()))
        results = []
        with backpack(ext.CMP(Curvature.GGN)):
            self.loss().backward()
            for p, mat in zip(self.model.parameters(), mat_list):
                results.append(p.cmp(mat))
        return results

    def kfra_blocks(self):
        results = []
        with backpack(ext.KFRA()):
            self.loss().backward()
            for p in self.model.parameters():
                factors = p.kfra
                results.append(matrix_from_kron_facs(factors))
        return results
