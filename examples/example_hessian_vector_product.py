"""
Compute the gradient and Hessian-vector products with PyTorch.
"""

import torch
from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential
from torch.nn.utils import parameters_to_vector

from backpack.hessianfree.hvp import hessian_vector_product
from backpack.utils.convert_parameters import vector_to_parameter_list
from utils import load_mnist_data

B = 64
X, y = load_mnist_data(B)

print("# Hessian-vector product and gradients with PyTorch | B =", B)

model = Sequential(Flatten(), Linear(784, 10),)
lossfunc = CrossEntropyLoss()

print("# 1) Vector with shapes like parameters | B =", B)

loss = lossfunc(model(X), y)
v = [torch.randn_like(p) for p in model.parameters()]

Hv = hessian_vector_product(loss, list(model.parameters()), v)

# has to be called afterwards, or with create_graph=True
loss.backward()

for (name, param), vec, Hvec in zip(model.named_parameters(), v, Hv):
    print(name)
    print(".grad.shape:                  ", param.grad.shape)
    # vector
    print("vector shape:                 ", vec.shape)
    # Hessian-vector product
    print("Hessian-vector product shape: ", Hvec.shape)

print("# 2) Flattened vector | B =", B)

loss = lossfunc(model(X), y)

num_params = sum(p.numel() for p in model.parameters())
v_flat = torch.randn(num_params)

v = vector_to_parameter_list(v_flat, model.parameters())
Hv = hessian_vector_product(loss, list(model.parameters()), v)
Hv_flat = parameters_to_vector(Hv)

# has to be called afterwards, or with create_graph=True
loss.backward()

print("Model parameters:                  ", num_params)
# vector
print("flat vector shape:                 ", v_flat.shape)
# individual gradient L2 norm
print("flat Hessian-vector product shape: ", Hv_flat.shape)
