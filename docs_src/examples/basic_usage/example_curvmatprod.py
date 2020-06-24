# TODO @sbharadwajj: Add some descriptive text, like here: ./example_all_in_one.py
# TODO @sbharadwajj: Add notes how to locally compile the docs/examples to dev guide

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

X, y = load_one_batch_mnist(batch_size=512)

model = Sequential(Flatten(), Linear(784, 20), Sigmoid(), Linear(20, 10))
lossfunc = CrossEntropyLoss()

model = extend(model)
lossfunc = extend(lossfunc)

loss = lossfunc(model(X), y)

with backpack(HMP(), GGNMP(), PCHMP()):
    loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print("type(.hmp):              ", type(param.hmp))
    print("type(.ggnmp):            ", type(param.ggnmp))
    print("type(.pchmp):            ", type(param.pchmp))

# multiply with a vector

# number of vectors
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

# multiply with a collection of vectors (a matrix)

# number of vectors
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
