"""
Compute the gradient with PyTorch and other quantities with BackPACK.
"""

from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential

from backpack import backpack, extend, extensions
from utils import load_mnist_data

B = 4
X, y = load_mnist_data(B)

print("# Gradient with PyTorch, other quantities with BackPACK | B =", B)

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
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    # individual gradients
    print(".grad_batch.shape:       ", param.grad_batch.shape)
    # gradient variance
    print(".variance.shape:         ", param.variance.shape)
    # gradient 2nd moment
    print(".sum_grad_squared.shape: ", param.sum_grad_squared.shape)
    # individual gradient L2 norm
    print(".batch_l2.shape:         ", param.batch_l2.shape)
    # MC-sampled GGN diagonal
    print(".diag_ggn_mc.shape:      ", param.diag_ggn_mc.shape)
    # Exact GGN diagonal
    print(".diag_ggn_exact.shape:   ", param.diag_ggn_exact.shape)
    # Exact Hessian diagonal
    print(".diag_h.shape:           ", param.diag_h.shape)
    # KFAC (Martens et al.)
    print(".kfac (shapes):          ", [kfac.shape for kfac in param.kfac])
    # KFLR (Botev et al.)
    print(".kflr (shapes):          ", [kflr.shape for kflr in param.kflr])
    # KFRA (Botev et al.)
    print(".kfra (shapes):          ", [kfra.shape for kfra in param.kfra])
