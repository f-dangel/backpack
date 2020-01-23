"""
Compute the full Hessian matrix with automatic differentiation.
Use Hessian-vector products for row-wise construction.
"""
import torch
from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential
from torch.nn.utils.convert_parameters import parameters_to_vector

from backpack.hessianfree.hvp import hessian_vector_product
from backpack.utils.convert_parameters import vector_to_parameter_list
from utils import load_mnist_data

B = 64
X, y = load_mnist_data(B)

print("# Hessian matrix with automatic differentiation | B =", B)

model = Sequential(Flatten(), Linear(784, 10),)
lossfunc = CrossEntropyLoss()

loss = lossfunc(model(X), y)

num_params = sum(p.numel() for p in model.parameters())
hessian = torch.zeros(num_params, num_params)

for i in range(num_params):
    # GGN-vector product with i.th unit vector yields the i.th row
    e_i = torch.zeros(num_params)
    e_i[i] = 1.0

    # convert to model parameter shapes
    e_i_list = vector_to_parameter_list(e_i, model.parameters())
    hessian_i_list = hessian_vector_product(loss, list(model.parameters()), e_i_list)

    hessian_i = parameters_to_vector(hessian_i_list)
    hessian[i, :] = hessian_i

print("Model parameters: ", num_params)
print("Hessian shape:    ", hessian.shape)
print("Hessian:          ", hessian)
