"""
Example using curvature-matrix products
=======================================

Basic example showing how to compute the gradient,
and to multiply with the block diagonal of

- the Hessian (``HMP``)
- the generalized Gauss-Newton (``GGNMP``)
- the positive-curvature Hessian (``PCHMP``)

for a single-hidden layer sigmoid network on MNIST.
"""

# %%
# Let's start by loading some dummy data and extending the model

from torch import rand
from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential, Sigmoid

from backpack import backpack, extend
from backpack.extensions import GGNMP, HMP, PCHMP
from backpack.utils.examples import load_one_batch_mnist

X, y = load_one_batch_mnist(batch_size=512)

model = Sequential(Flatten(), Linear(784, 20), Sigmoid(), Linear(20, 10))
lossfunc = CrossEntropyLoss()

model = extend(model)
lossfunc = extend(lossfunc)


# %%
# Let's inspect the objects that are created during the backward pass with
# BackPACK's curvature-matrix product (``*MP``) extensions:
# Functions that multiply with the associated curvature matrix.

loss = lossfunc(model(X), y)

with backpack(HMP(), GGNMP(), PCHMP()):
    loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:               ", param.grad.shape)
    print("type(.hmp):                ", type(param.hmp))
    print("type(.ggnmp):              ", type(param.ggnmp))
    print("type(.pchmp):              ", type(param.pchmp))

# %%
# Let's multiply a random vector with different curvature blocks.

num_vecs = 1

for name, param in model.named_parameters():
    vec = rand(num_vecs, *param.shape)
    print(name)
    print(".grad.shape:              ", param.grad.shape)
    print("vec.shape:                ", vec.shape)
    print(".hmp(vec).shape:          ", param.hmp(vec).shape)
    print(".ggnmp(vec).shape:        ", param.ggnmp(vec).shape)
    print(".pchmp(vec, 'abs').shape: ", param.pchmp(vec, modify="abs").shape)
    print(".pchmp(vec, 'clip').shape:", param.pchmp(vec, modify="clip").shape)
    print("*"*50)

# %%
# We can also multiply a collection of vectors (a matrix) at once.
num_vecs = 3

for name, param in model.named_parameters():
    vec = rand(num_vecs, *param.shape)
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print("vec.shape:               ", vec.shape)
    print(".hmp(vec).shape:          ", param.hmp(vec).shape)
    print(".ggnmp(vec).shape:        ", param.ggnmp(vec).shape)
    print(".pchmp(vec, 'abs').shape: ", param.pchmp(vec, modify="abs").shape)
    print(".pchmp(vec, 'clip').shape:", param.pchmp(vec, modify="clip").shape)
    print("*"*50)
