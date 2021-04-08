"""Introduction for Tim."""
import time
import torch
from backpack import extend
from backpack.core.derivatives.linear import LinearDerivatives
torch.manual_seed(0)
# create fake data
N = 128
D_in = 784
D_out = 64
inputs = torch.rand(N, D_in)
inputs.requires_grad = True
# print(inputs.shape)
# backpropagate with torch.autograd
layer = torch.nn.Linear(D_in, D_out, bias=False)
outputs = layer(inputs)
# print(outputs.shape)
# print(layer.weight.shape)
grad_outputs = torch.rand_like(outputs)
grad_inputs_autograd = torch.autograd.grad(
    [outputs],
    [inputs],
    grad_outputs=[grad_outputs],
)[0]
# print(grad_inputs_autograd)
# backpropagate with BackPACK
derivatives = LinearDerivatives()
layer = extend(layer)
outputs = layer(inputs)
grad_inputs_backpack = derivatives.jac_t_mat_prod(layer, None, None, grad_outputs)
# print(torch.allclose(grad_inputs_autograd, grad_inputs_backpack))
# summed gradient w.r.t. weight
outputs = layer(inputs)
grad_weight = torch.autograd.grad(
    [outputs], [layer.weight], grad_outputs=[grad_outputs]
)[0]
# print(grad_weight)
# print(grad_weight.shape, layer.weight.shape)
# individual gradients for weights with torch.autograd
igrad_autograd = torch.zeros(N, *layer.weight.shape)
# print(igrad_autograd.shape)
start = time.time()
for n in range(N):
    inputs_n = inputs[n]
    outputs_n = layer(inputs_n)
    grad_outputs_n = grad_outputs[n]
    grad_weight_n = torch.autograd.grad(
        [outputs_n], [layer.weight], grad_outputs=[grad_outputs_n]
    )[0]
    igrad_autograd[n] = grad_weight_n
end = time.time()
print(f"Autograd: {end - start}")
# individual gradients for weights with BackPACK
start = time.time()
outputs = layer(inputs)
igrad_backpack = derivatives.weight_jac_t_mat_prod(
    layer, None, None, grad_outputs, sum_batch=False
)
end = time.time()
# print(igrad_backpack.shape)
print(f"BackPACK: {end - start}")
print("BackPACK = autograd?: ", torch.allclose(igrad_autograd, igrad_backpack))
