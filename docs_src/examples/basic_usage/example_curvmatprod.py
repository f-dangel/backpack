"""
Example using curvature-matrix products
=======================================

Basic example showing how compute the gradient,
and to multiply with the block-diagonal of

- the Hessian
- the generalized Gauss-Newton
- the positive-curvature Hessian

on a 2-layer network for MNIST with sigmoid activation.
For this network, all of the above curvatures are different.
"""


from torch import rand
from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential, Sigmoid

from backpack import backpack, extend
from backpack.extensions import GGNMP, HMP, PCHMP
from backpack.utils.examples import load_one_batch_mnist

# %%
# We first create a simple sequential model and load MNIST data
X, y = load_one_batch_mnist(batch_size=512)

model = Sequential(Flatten(), Linear(784, 20), Sigmoid(), Linear(20, 10))
lossfunc = CrossEntropyLoss()

model = extend(model)
lossfunc = extend(lossfunc)


# %%
# Hessian matrix product
loss = lossfunc(model(X), y)

with backpack(HMP()):
    loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print("type(.hmp):              ", type(param.hmp))

# %%
# Generalized Gauss Newton Matrix Product 
loss = lossfunc(model(X), y)

with backpack(GGNMP()):
    loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print("type(.ggnmp):              ", type(param.ggnmp))

# %%
# The Positive Curvature Hessian

loss = lossfunc(model(X), y)

with backpack(PCHMP()):
    loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print("type(.pchmp):              ", type(param.pchmp))



# %% 
# Here we multiply the curvature methods with vectors and we first show it for a single vector. Its also possible to ask for multiple quantitites at once

V = 1

for name, param in model.named_parameters():
    vec = rand(V, *param.shape)
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print("vec.shape:               ", vec.shape)
    print("hmp(vec).shape:          ", param.hmp(vec).shape)
    print("ggnmp(vec).shape:        ", param.ggnmp(vec).shape)
    print("pchmp(vec, 'abs').shape: ", param.pchmp(vec, modify="abs").shape)
    print("pchmp(vec, 'clip').shape:", param.pchmp(vec, modify="clip").shape)
    print("*"*50)

# %%
# We can also multiply multiple vectors at once and still have a inexpensive computational cost.
V = 3

for name, param in model.named_parameters():
    vec = rand(V, *param.shape)
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print("vec.shape:               ", vec.shape)
    print("hmp(vec).shape:          ", param.hmp(vec).shape)
    print("ggnmp(vec).shape:        ", param.ggnmp(vec).shape)
    print("pchmp(vec, 'abs').shape: ", param.pchmp(vec, modify="abs").shape)
    print("pchmp(vec, 'clip').shape:", param.pchmp(vec, modify="clip").shape)
    print("*"*50)