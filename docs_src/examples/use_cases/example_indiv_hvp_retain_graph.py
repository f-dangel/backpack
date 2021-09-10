"""Individual Hessian-Vector-Product and retain_graph
======================================================
"""
# %%
# imports
import torch
from torch import nn
from torch.autograd import grad

from backpack import backpack, extend
from backpack.extensions import BatchGrad
from backpack.utils.examples import load_one_batch_mnist

BATCH_SIZE = 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


# %%
# Let's define a function that calculates individual hessian_vector products.
# This function was provided by Haonan Wang (github: haonan3) in #213.
def batch_hvp(model, loss, params_list, batch_grad_list):
    if len(params_list) != len(batch_grad_list):
        raise (ValueError("w and v must have the same length."))

    with backpack(retain_graph=True):
        one_sample_grad_list = grad(
            loss, params_list, retain_graph=True, create_graph=True
        )

    elemwise_products = 0.0
    for grad_elem, v_elem in zip(one_sample_grad_list, batch_grad_list):
        sum_over_dims = []
        for i in range(len(v_elem.shape)):
            sum_over_dims.append(i)
        sum_over_dims = tuple(sum_over_dims[1:])
        elemwise_products += torch.sum(
            grad_elem.unsqueeze(0) * v_elem.detach(), sum_over_dims
        )

    with backpack(BatchGrad()):
        elemwise_products.backward()  # problem: has no attribute 'input0'
        return_grads = [p.grad_batch for p in model.parameters() if p.requires_grad]

    return return_grads


# %%
# create the model and do one forward pass
def make_model():
    return nn.Sequential(
        nn.Conv2d(1, 10, 5, 1),
        nn.Sigmoid(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(10, 20, 5, 1),
        nn.Sigmoid(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(4 * 4 * 20, 50),
        nn.Sigmoid(),
        nn.Linear(50, 10),
    )


model = make_model().to(DEVICE)
loss_function = torch.nn.CrossEntropyLoss().to(DEVICE)
model = extend(model)
loss_function = extend(loss_function)

x, y = load_one_batch_mnist(BATCH_SIZE)
x, y = x.to(DEVICE), y.to(DEVICE)
loss = loss_function(model(x), y)

# %%
# Create a random vector and apply the batch_hvp function.
params_list = list(model.parameters())
batch_grad_list = [torch.rand(1, *param.shape, device=DEVICE) for param in params_list]

batch_hvp = batch_hvp(model, loss, params_list, batch_grad_list)

print("\nresult")
for result in batch_hvp:
    print(result.shape)


# %%
# Compare the result with the for-loop version
def batch_hvp_for_loop(model, loss, params_list, batch_grad_list):
    if len(params_list) != len(batch_grad_list):
        raise (ValueError("w and v must have the same length."))

    one_sample_grad_list = grad(loss, params_list, retain_graph=True, create_graph=True)

    elemwise_products = 0
    for grad_elem, v_elem in zip(one_sample_grad_list, batch_grad_list):
        sum_over_dims = []
        for i in range(len(v_elem.shape)):
            sum_over_dims.append(i)
        sum_over_dims = tuple(sum_over_dims[1:])
        elemwise_products += torch.sum(
            grad_elem.unsqueeze(0) * v_elem.detach(), sum_over_dims
        )

    # The for-loop version
    grad_cache = []
    for i in range(elemwise_products.shape[0]):
        elemwise_products[i].backward(retain_graph=True)
        grad_cache.append(
            [p.grad.clone() for p in model.parameters() if p.requires_grad]
        )
    grad_cache = list(zip(*grad_cache))
    return_grads = []
    for l_id in range(len(grad_cache)):
        return_grads.append(
            torch.cat([g.unsqueeze(0) for g in grad_cache[l_id]], dim=0)
        )

    return return_grads


model.zero_grad()
loss = loss_function(model(x), y)
batch_hvp_for_loop = batch_hvp_for_loop(model, loss, params_list, batch_grad_list)

print("\nresult for loop")
for result, result_compare in zip(batch_hvp, batch_hvp_for_loop):
    match = torch.allclose(result.sum(0).unsqueeze(0), result_compare, atol=1e-3)
    print(match)
    if match is False:
        print(result.shape)
        print(result_compare.shape)
        # print(result[:, 0, 0, 0, :].sum(0))
        # print(result_compare[0, 0, 0, 0, :])
        # raise AssertionError("Results don't match.")
