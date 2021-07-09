import torch

from backpack import extend

DEVICE_CPU = torch.device("cpu")


class TestProblem:
    def __init__(self, X, Y, model, lossfunc, device=DEVICE_CPU):
        """
        A traditional machine learning test problem, loss(model(X), Y)

        X: [N x D_X]
        Y: [N x D_Y]
        model: [N x D_X] -> [N x D_out]
        loss: [N x D_out] x [N x D_y] -> scalar
        """
        self.X = X
        self.Y = Y
        self.model = extend(model)
        self.lossfunc = extend(lossfunc)
        self.device = device
        self.to(device)
        self.N = self.X.shape[0]

    def to(self, device):
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        self.model = self.model.to(device)
        self.lossfunc = self.lossfunc.to(device)
        self.device = device
        return self

    def loss(self, b=None):
        """
        Computes the loss for the input-outputs.
        If b, a sample ID, is set, computes the loss for that sample
        """
        if b is None:
            return self.lossfunc(self.model(self.X), self.Y)
        else:
            Xb = self.X[b, :].unsqueeze(0)
            if len(self.Y.shape) > 1:
                Yb = self.Y[b, :].unsqueeze(0)
            else:
                Yb = self.Y[b].unsqueeze(0)
            return self.lossfunc(self.model(Xb), Yb)

    def clear(self):
        """
        Clear saved state
        """
        attrs = ["sum_grad_squared" "grad_batch" "grad" "diag_ggn"]

        def safeclear(p, attr):
            if hasattr(p, attr):
                delattr(p, attr)

        for p in self.model.parameters():
            for attr in attrs:
                safeclear(p, attr)
