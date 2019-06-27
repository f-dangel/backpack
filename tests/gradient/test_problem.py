import torch
import pytest
import numpy
import unittest
from bpexts.utils import set_seeds
import bpexts.gradient.config as config
import bpexts.hessian.free as HF
import bpexts.hessian.exact as exact
import bpexts.utils as utils
from bpexts.gradient.extensions import Extensions as ext


class TestProblem():

    def __init__(self, X, Y, model, loss, device=torch.device("cpu")):
        """
        A traditional machine learning test problem, loss(model(X), Y)

        X: [N x D_X]
        Y: [N x D_Y]
        model: [N x D_X] -> [N x D_out]
        loss: [N x D_out] x [N x D_y] -> scalar
        """
        self.X = X.to(device)
        self.Y = Y.to(device)
        self.model = model.to(device)
        self.loss = loss
        self.device = device
        self.N = self.X.shape[0]

    def compute_loss(self, b=None):
        """
        Computes the loss for the input-outputs.
        If b, a sample ID, is set, computes the loss for that sample
        """
        if b is None:
            return self.loss(self.model(self.X), self.Y)
        else:
            return self.loss(self.model(self.X[b, :].unsqueeze(0)), self.Y[b, :].unsqueeze(0))

    def batch_gradients_autograd(self):
        batch_grads = [
            torch.zeros(self.N, *p.size()).to(self.device)
            for p in self.model.parameters()
        ]
        for b in range(self.N):
            gradients = torch.autograd.grad(self.compute_loss(b), self.model.parameters())
            for idx, g in enumerate(gradients):
                batch_grads[idx][b, :] = g.detach() / self.N
        return batch_grads

    def batch_gradients_bpexts(self):
        with config.bpexts(ext.BATCH_GRAD):
            self.compute_loss().backward()

            batch_grads = []
            for p in self.model.parameters():
                batch_grads.append(p.grad_batch)
        return batch_grads

    def gradient_autograd(self):
        return list(torch.autograd.grad(self.compute_loss(), self.model.parameters()))

    def sgs_autograd(self):
        """Sum of squared gradients via torch.autograd."""
        batch_grad = self._compute_batch_gradients_autograd()
        sgs = [(g**2).sum(0) for g in batch_grad]
        return sgs

    def sgs_bpexts(self):
        """Sum of squared gradients via bpexts."""
        with config.bpexts(ext.SUM_GRAD_SQUARED):
            self.compute_loss().backward()
            sgs = [p.sum_grad_squared for p in self.model.parameters()]
        return sgs

    def diag_ggn_autograd(self):
        """Diagonal of the GGN matrix via autodiff."""

        outputs = self.model(self.X)
        loss = self.loss(outputs, self.Y)

        tot_params = sum([p.numel() for p in self.model.parameters()])
        i = 0
        diag_ggns = []

        for p in list(self.model.parameters()):
            diag_ggn_p = torch.zeros_like(p).view(-1)
            # extract i.th column of the GGN, store diagonal entry
            for i_p in range(p.numel()):
                v = torch.zeros(tot_params).to(self.device)
                v[i] = 1.

                vs = HF.vector_to_parameter_list(
                    v, list(self.model.parameters()))

                # GGN-vector product
                GGN_v = HF.ggn_vector_product(
                    loss, outputs, self.model, vs)
                GGN_v = torch.cat([g.detach().view(-1) for g in GGN_v])

                diag_ggn_p[i_p] = GGN_v[i]
                i += 1
            # reshape into parameter dimension and append
            diag_ggns.append(diag_ggn_p.view(p.size()))
        return diag_ggns

        def _compute_diag_ggn_bpexts(self):
            """Diagonal of the GGN matrix via bpexts."""
            with config.bpexts(ext.DIAG_GGN):
                self.compute_loss().backward()
                diag_ggns = [p.diag_ggn for p in self.model.parameters()]
            return diag_ggns

        def _compute_loss_hessian_autograd(self):
            """Compute the Hessian of the individual loss w.r.t. the output.

            Return a tensor storing the loss Hessians for each sample along
            the first dimension.
            """
            layer = self._create_layer()
            inputs = self._create_input()
            loss_hessians = []
            for b in range(inputs.size(0)):
                input = inputs[b, :].unsqueeze(0)
                output = layer(input)
                loss = self._loss_fn(output) / inputs.size(0)

                h = exact.exact_hessian(loss, [output]).detach()
                loss_hessians.append(h.detach())

            loss_hessians = torch.stack(loss_hessians)
            assert tuple(loss_hessians.size()) == (inputs.size(0),
                                                   output.numel(),
                                                   output.numel())
            return loss_hessians

    def clear(self):
        """
        Clear saved state
        """
        attrs = [
            "sum_grad_squared"
            "grad_batch"
            "grad"
            "diag_ggn"
        ]

        def safeclear(p, attr):
            if hasattr(p, attr):
                delattr(p, attr)

        for p in self.model.parameters():
            for attr in attrs:
                safeclear(p, attr)
