"""
Compute the gradient with PyTorch and other quantities with BackPACK.
"""

from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential

from backpack import backpack, extend, extensions
from utils import load_mnist_data

B = 64
X, y = load_mnist_data(B)

print(f"# Gradient with PyTorch, other quantities with BackPACK (B={B})")

model = Sequential(
    Flatten(),
    Linear(784, 10),
)
lossfunc = CrossEntropyLoss()

model = extend(model)
lossfunc = extend(lossfunc)

loss = lossfunc(model(X), y)

with backpack(
        # individual gradients
        extensions.BatchGrad(),
        # gradient variance
        extensions.Variance(),
        # gradient 2nd moment
        extensions.SumGradSquared(),
        # individual gradient L2 norm
        extensions.BatchL2Grad(),
        # MC-sampled GGN diagonal
        extensions.DiagGGNMC(),
        # Exact GGN diagonal
        extensions.DiagGGNExact(),
        # Exact Hessian diagonal
        extensions.DiagHessian(),
        # KFAC (Martens et al.)
        extensions.KFAC(),
        # KFLR (Botev et al.)
        extensions.KFLR(),
        # KFRA (Botev et al.)
        extensions.KFRA(),
):
    loss.backward()

for name, param in model.named_parameters():
    print(
        f"\n{name}:",
        # gradient
        f"\n\t.grad.shape:             {param.grad.shape}",
        # individual gradients
        f"\n\t.grad_batch.shape:       {param.grad_batch.shape}",
        # gradient variance
        f"\n\t.variance.shape:         {param.variance.shape}",
        # gradient 2nd moment
        f"\n\t.sum_grad_squared.shape: {param.sum_grad_squared.shape}",
        # individual gradient L2 norm
        f"\n\t.batch_l2.shape:         {param.batch_l2.shape}",
        # MC-sampled GGN diagonal
        f"\n\t.diag_ggn_mc.shape:      {param.diag_ggn_mc.shape}",
        # Exact GGN diagonal
        f"\n\t.diag_ggn_exact.shape:   {param.diag_ggn_exact.shape}",
        # Exact Hessian diagonal
        f"\n\t.diag_h.shape:           {param.diag_h.shape}",
        # KFAC (Martens et al.)
        f"\n\t.kfac (shapes):          {[kfac.shape for kfac in param.kfac]}",
        # KFLR (Botev et al.)
        f"\n\t.kflr (shapes):          {[kflr.shape for kflr in param.kflr]}",
        # KFRA (Botev et al.)
        f"\n\t.kfra (shapes):          {[kfra.shape for kfra in param.kfra]}",
    )
