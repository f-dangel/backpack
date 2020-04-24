"""
Compute the full GGN matrix with automatic differentiation.
Use GGN-vector products for row-wise construction.
"""
import torch
from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential
from torch.nn.utils.convert_parameters import parameters_to_vector

from backpack.hessianfree.ggnvp import ggn_vector_product
from backpack.utils.convert_parameters import vector_to_parameter_list
from backpack.utils.examples import load_one_batch_mnist

B = 4
X, y = load_one_batch_mnist(B)

print("# GGN matrix with automatic differentiation | B =", B)

model = Sequential(Flatten(), Linear(784, 10),)
lossfunc = CrossEntropyLoss()

output = model(X)
loss = lossfunc(output, y)

num_params = sum(p.numel() for p in model.parameters())
ggn = torch.zeros(num_params, num_params)

for i in range(num_params):
    # GGN-vector product with i.th unit vector yields the i.th row
    e_i = torch.zeros(num_params)
    e_i[i] = 1.0

    # convert to model parameter shapes
    e_i_list = vector_to_parameter_list(e_i, model.parameters())
    ggn_i_list = ggn_vector_product(loss, output, model, e_i_list)

    ggn_i = parameters_to_vector(ggn_i_list)
    ggn[i, :] = ggn_i

print("Model parameters: ", num_params)
print("GGN shape:        ", ggn.shape)
print("GGN:              ", ggn)
