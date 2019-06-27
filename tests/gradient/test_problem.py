import torch
import bpexts.gradient.config as config
import bpexts.hessian.free as HF
from bpexts.gradient.extensions import Extensions as ext


class TestProblem():
    """
    Given a ML problem (fitting X to Y using model under lossfunc),
    compute various quantities at the current parameters using brute-force
    autodiff or bpexts.
    """

    def __init__(self, X, Y, model, lossfunc, device=torch.device("cpu")):
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
        self.lossfunc = lossfunc
        self.device = device
        self.N = self.X.shape[0]

    def loss(self, b=None):
        """
        Computes the loss for the input-outputs.
        If b, a sample ID, is set, computes the loss for that sample
        """
        if b is None:
            return self.lossfunc(self.model(self.X), self.Y)
        else:
            Xb, Yb = self.X[b, :].unsqueeze(0), self.Y[b, :].unsqueeze(0)
            return self.lossfunc(self.model(Xb), Yb)

    def gradient_autograd(self):
        return list(torch.autograd.grad(self.loss(), self.model.parameters()))

    def batch_gradients_autograd(self):
        batch_grads = [
            torch.zeros(self.N, *p.size()).to(self.device)
            for p in self.model.parameters()
        ]

        for b in range(self.N):
            gradients = torch.autograd.grad(self.loss(b), self.model.parameters())

            for idx, g in enumerate(gradients):
                batch_grads[idx][b, :] = g.detach() / self.N

        return batch_grads

    def batch_gradients_bpexts(self):
        with config.bpexts(ext.BATCH_GRAD):
            self.loss().backward()

            batch_grads = []
            for p in self.model.parameters():
                batch_grads.append(p.grad_batch)

        return batch_grads

    def sgs_autograd(self):
        batch_grad = self.batch_gradients_autograd()
        sgs = [(g**2).sum(0) for g in batch_grad]
        return sgs

    def sgs_bpexts(self):
        with config.bpexts(ext.SUM_GRAD_SQUARED):
            self.loss().backward()
            sgs = [p.sum_grad_squared for p in self.model.parameters()]
        return sgs

    def diag_ggn_autograd(self):
        outputs = self.model(self.X)
        loss = self.lossfunc(outputs, self.Y)

        tot_params = sum([p.numel() for p in self.model.parameters()])

        def extract_ith_element_of_diag_ggn(i):
            v = torch.zeros(tot_params).to(self.device)
            v[i] = 1.

            vs = HF.vector_to_parameter_list(
                v, list(self.model.parameters()))

            # GGN-vector product
            GGN_v = HF.ggn_vector_product(loss, outputs, self.model, vs)
            GGN_v = torch.cat([g.detach().view(-1) for g in GGN_v])
            return GGN_v[i]

        diagonal_index = 0
        diag_ggns = []
        for p in list(self.model.parameters()):
            diag_ggn_p = torch.zeros_like(p).view(-1)

            for parameter_index in range(p.numel()):
                diag_value = extract_ith_element_of_diag_ggn(diagonal_index)
                diag_ggn_p[parameter_index] = diag_value
                diagonal_index += 1

            diag_ggns.append(diag_ggn_p.view(p.size()))

        return diag_ggns

    def diag_ggn_bpexts(self):
        with config.bpexts(ext.DIAG_GGN):
            self.loss().backward()
            diag_ggns = [p.diag_ggn for p in self.model.parameters()]
        return diag_ggns

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


def loss0(x, y=None):
    """Dummy loss function: Normalized sum of squared elements."""
    return (x**2).contiguous().view(x.size(0), -1).mean(0).sum()


def loss1(x, y=None):
    loss = torch.zeros(1).to(x.device)
    for b in range(x.size(0)):
        loss += (x[b, :].view(-1).sum())**2 / x.size(0)
    return loss


def loss2(x, y=None):
    loss = torch.zeros(1).to(x.device)
    for b in range(x.size(0)):
        loss += (
            torch.log10(torch.abs(x[b, :]) + 0.1).sum())**2 / x.size(0)
    return loss


losses = [loss0, loss1, loss2]
