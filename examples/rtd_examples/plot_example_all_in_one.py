"""
All in one
===========================

Compute the gradient with PyTorch and other quantities with BackPACK.
"""

from backpack.utils.examples import load_mnist_data
from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential
from backpack import backpack, extend, extensions

# %%
# Let's start by loading some dummy data
# --------------------------------------

B = 4
X, y = load_mnist_data(B)

print("# Gradient with PyTorch, other quantities with BackPACK | B =", B)

# %%
# Create a model and loss function
# --------------------------------

model = Sequential(Flatten(), Linear(784, 10),)
lossfunc = CrossEntropyLoss()

model = extend(model)
lossfunc = extend(lossfunc)

# %%
# We can now evaluate the loss and do a backward pass with Backpack
# -----------------------------------------------------------------

loss = lossfunc(model(X), y)

with backpack(
    extensions.BatchGrad(),
    extensions.Variance(),
    extensions.SumGradSquared(),
    extensions.BatchL2Grad(),
    extensions.DiagGGNMC(mc_samples=1),
    extensions.DiagGGNExact(),
    extensions.DiagHessian(),
    extensions.KFAC(mc_samples=1),
    extensions.KFLR(),
    extensions.KFRA(),
):
    loss.backward()

# %%
# And here are the results
# -----------------------------------------------------------------

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print(".grad_batch.shape:       ", param.grad_batch.shape)
    print(".variance.shape:         ", param.variance.shape)
    print(".sum_grad_squared.shape: ", param.sum_grad_squared.shape)
    print(".batch_l2.shape:         ", param.batch_l2.shape)
    print(".diag_ggn_mc.shape:      ", param.diag_ggn_mc.shape)
    print(".diag_ggn_exact.shape:   ", param.diag_ggn_exact.shape)
    print(".diag_h.shape:           ", param.diag_h.shape)
    print(".kfac (shapes):          ", [kfac.shape for kfac in param.kfac])
    print(".kflr (shapes):          ", [kflr.shape for kflr in param.kflr])
    print(".kfra (shapes):          ", [kfra.shape for kfra in param.kfra])
