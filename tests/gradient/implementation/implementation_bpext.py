import torch
from .implementation import Implementation
from bpexts.gradient import bpexts
from bpexts.gradient.extensions import Extensions as ext


class BpextImpl(Implementation):

    def gradient(self):
        return list(torch.autograd.grad(self.loss(), self.model.parameters()))

    def batch_gradients(self):
        with bpexts(ext.BATCH_GRAD):
            self.loss().backward()
            batch_grads = [p.grad_batch for p in self.model.parameters()]
        return batch_grads

    def batch_l2(self):
        with bpexts(ext.BATCH_L2):
            self.loss().backward()
            batch_l2s = [p.batch_l2 for p in self.model.parameters()]
        return batch_l2s

    def variance(self):
        with bpexts(ext.VARIANCE):
            self.loss().backward()
            variances = [p.variance for p in self.model.parameters()]
        return variances

    def sgs(self):
        with bpexts(ext.SUM_GRAD_SQUARED):
            self.loss().backward()
            sgs = [p.sum_grad_squared for p in self.model.parameters()]
        return sgs

    def diag_ggn(self):
        with bpexts(ext.DIAG_GGN):
            self.loss().backward()
            diag_ggns = [p.diag_ggn for p in self.model.parameters()]
        return diag_ggns
