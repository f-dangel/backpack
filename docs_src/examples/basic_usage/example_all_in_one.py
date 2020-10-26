"""
Example using all extensions
==============================

Basic example showing how compute the gradient,
and and other quantities with BackPACK,
on a linear model for MNIST.
"""

# %%
# Let's start by loading some dummy data and extending the model

from torch import rand
from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential

from backpack import backpack, extend
from backpack.extensions import (
    GGNMP,
    HMP,
    KFAC,
    KFLR,
    KFRA,
    PCHMP,
    BatchGrad,
    BatchL2Grad,
    DiagGGNExact,
    DiagGGNMC,
    DiagHessian,
    SumGradSquared,
    Variance,
)
from backpack.utils.examples import load_one_batch_mnist

X, y = load_one_batch_mnist(batch_size=512)

model = Sequential(Flatten(), Linear(784, 10),)
lossfunc = CrossEntropyLoss()

model = extend(model)
lossfunc = extend(lossfunc)

# %%
# First order extensions
# ----------------------

# %%
# Batch gradients

loss = lossfunc(model(X), y)
with backpack(BatchGrad()):
    loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print(".grad_batch.shape:       ", param.grad_batch.shape)

# %%
# Variance

loss = lossfunc(model(X), y)
with backpack(Variance()):
    loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print(".variance.shape:         ", param.variance.shape)

# %%
# Second moment/sum of gradients squared

loss = lossfunc(model(X), y)
with backpack(SumGradSquared()):
    loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print(".sum_grad_squared.shape: ", param.sum_grad_squared.shape)

# %%
# L2 norm of individual gradients

loss = lossfunc(model(X), y)
with backpack(BatchL2Grad()):
    loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print(".batch_l2.shape:         ", param.batch_l2.shape)

# %%
# It's also possible to ask for multiple quantities at once

loss = lossfunc(model(X), y)
with backpack(BatchGrad(), Variance(), SumGradSquared(), BatchL2Grad()):
    loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print(".grad_batch.shape:       ", param.grad_batch.shape)
    print(".variance.shape:         ", param.variance.shape)
    print(".sum_grad_squared.shape: ", param.sum_grad_squared.shape)
    print(".batch_l2.shape:         ", param.batch_l2.shape)

# %%
# Second order extensions
# --------------------------

# %%
# Diagonal of the Gauss-Newton and its Monte-Carlo approximation

loss = lossfunc(model(X), y)
with backpack(DiagGGNExact(), DiagGGNMC(mc_samples=1)):
    loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print(".diag_ggn_mc.shape:      ", param.diag_ggn_mc.shape)
    print(".diag_ggn_exact.shape:   ", param.diag_ggn_exact.shape)

# %%
# KFAC, KFRA and KFLR

loss = lossfunc(model(X), y)
with backpack(KFAC(mc_samples=1), KFLR(), KFRA()):
    loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print(".kfac (shapes):          ", [kfac.shape for kfac in param.kfac])
    print(".kflr (shapes):          ", [kflr.shape for kflr in param.kflr])
    print(".kfra (shapes):          ", [kfra.shape for kfra in param.kfra])

# %%
# Diagonal Hessian

loss = lossfunc(model(X), y)
with backpack(DiagHessian()):
    loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print(".diag_h.shape:           ", param.diag_h.shape)

# %%
# Block-diagonal curvature products
# ---------------------------------

# %%
# Curvature-matrix product (``MP``) extensions provide functions
# that multiply with the block diagonal of different curvature matrices, such as
#
# - the Hessian (:code:`HMP`)
# - the generalized Gauss-Newton (:code:`GGNMP`)
# - the positive-curvature Hessian (:code:`PCHMP`)

loss = lossfunc(model(X), y)

with backpack(
    HMP(),
    GGNMP(),
    PCHMP(savefield="pchmp_clip", modify="clip"),
    PCHMP(savefield="pchmp_abs", modify="abs"),
):
    loss.backward()

# %%
# Multiply a random vector with curvature blocks.

V = 1

for name, param in model.named_parameters():
    vec = rand(V, *param.shape)
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print("vec.shape:               ", vec.shape)
    print(".hmp(vec).shape:         ", param.hmp(vec).shape)
    print(".ggnmp(vec).shape:       ", param.ggnmp(vec).shape)
    print(".pchmp_clip(vec).shape:  ", param.pchmp_clip(vec).shape)
    print(".pchmp_abs(vec).shape:   ", param.pchmp_abs(vec).shape)

# %%
# Multiply a collection of three vectors (a matrix) with curvature blocks.

V = 3

for name, param in model.named_parameters():
    vec = rand(V, *param.shape)
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print("vec.shape:               ", vec.shape)
    print(".hmp(vec).shape:         ", param.hmp(vec).shape)
    print(".ggnmp(vec).shape:       ", param.ggnmp(vec).shape)
    print(".pchmp_clip(vec).shape:  ", param.pchmp_clip(vec).shape)
    print(".pchmp_abs(vec).shape:   ", param.pchmp_abs(vec).shape)
