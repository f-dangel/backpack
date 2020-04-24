"""
Compute the gradient with PyTorch and other quantities with BackPACK.

CNN example with many different layers.
"""


from torch.nn import (
    AvgPool2d,
    Conv2d,
    CrossEntropyLoss,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)

from backpack import backpack, extend, extensions
from backpack.utils.examples import load_mnist_one_batch

B = 4
X, y = load_mnist_one_batch(B)

print("# Gradient with PyTorch, other quantities with BackPACK | B =", B)

model = Sequential(
    Conv2d(in_channels=1, out_channels=8, kernel_size=5),
    ReLU(),
    Conv2d(in_channels=8, out_channels=8, kernel_size=5),
    MaxPool2d(kernel_size=2),
    Sigmoid(),
    Conv2d(in_channels=8, out_channels=16, kernel_size=5),
    AvgPool2d(kernel_size=2),
    Dropout(p=0.5),
    Flatten(),
    Linear(3 * 3 * 16, 64),
    Tanh(),
    Linear(64, 10),
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
    # number of samples optional (default: 1)
    extensions.DiagGGNMC(mc_samples=1),
    # Exact GGN diagonal
    extensions.DiagGGNExact(),
    # Exact Hessian diagonal
    extensions.DiagHessian(),
    # KFAC (Martens et al.)
    # number of samples optional (default: 1)
    extensions.KFAC(mc_samples=1),
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
