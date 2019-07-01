import torch
from .implementation import Implementation
import bpexts.gradient.config as config
from bpexts.gradient.extensions import Extensions as ext


class BpextImpl(Implementation):

    def gradient(self):
        return list(torch.autograd.grad(self.loss(), self.model.parameters()))

    def batch_gradients(self):
        with config.bpexts(ext.BATCH_GRAD):
            self.loss().backward()

            batch_grads = []
            for p in self.model.parameters():
                batch_grads.append(p.grad_batch)

        return batch_grads

    def sgs(self):
        with config.bpexts(ext.SUM_GRAD_SQUARED):
            self.loss().backward()
            sgs = [p.sum_grad_squared for p in self.model.parameters()]
        return sgs

    def diag_ggn(self):
        with config.bpexts(ext.DIAG_GGN):
            self.loss().backward()
            diag_ggns = [p.diag_ggn for p in self.model.parameters()]
        return diag_ggns
