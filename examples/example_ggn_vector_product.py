"""
Compute the gradient and Hessian-vector products with PyTorch.
"""

import torch
from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential
from torch.nn.utils import parameters_to_vector

from backpack.hessianfree.ggnvp import ggn_vector_product
from backpack.utils.convert_parameters import vector_to_parameter_list
from backpack.utils.examples import load_mnist_one_batch

B = 4
X, y = load_mnist_one_batch(B)

print("# GGN-vector product and gradients with PyTorch | B =", B)

model = Sequential(Flatten(), Linear(784, 10),)
lossfunc = CrossEntropyLoss()

print("# 1) Vector with shapes like parameters | B =", B)

output = model(X)
loss = lossfunc(output, y)
v = [torch.randn_like(p) for p in model.parameters()]

GGNv = ggn_vector_product(loss, output, model, v)

# has to be called afterwards, or with create_graph=True
loss.backward()

for (name, param), vec, GGNvec in zip(model.named_parameters(), v, GGNv):
    print(name)
    print(".grad.shape:              ", param.grad.shape)
    # vector
    print("vector shape:             ", vec.shape)
    # Hessian-vector product
    print("GGN-vector product shape: ", GGNvec.shape)

print("# 2) Flattened vector | B =", B)

output = model(X)
loss = lossfunc(output, y)

num_params = sum(p.numel() for p in model.parameters())
v_flat = torch.randn(num_params)

v = vector_to_parameter_list(v_flat, model.parameters())
GGNv = ggn_vector_product(loss, output, model, v)
GGNv_flat = parameters_to_vector(GGNv)

# has to be called afterwards, or with create_graph=True
loss.backward()

print("Model parameters:              ", num_params)
# vector
print("flat vector shape:             ", v_flat.shape)
# individual gradient L2 norm
print("flat GGN-vector product shape: ", GGNv_flat.shape)
