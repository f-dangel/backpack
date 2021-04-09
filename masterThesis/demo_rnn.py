import torch
from backpack import extend

torch.manual_seed(0)

# create fake data
N = 1  # batch size
seq_len = 4
input_size = 5
hidden_size = 3
inputs = torch.rand(seq_len, N, input_size)
inputs.requires_grad = True
print("inputs.shape", inputs.shape)

'''
backpropagation with torch.autograd 
'''
layer = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1)
outputs, _ = layer(inputs)
print("outputs.shape", outputs.shape)
grad_outputs = torch.rand_like(outputs)
print("grad_outputs.shape", grad_outputs.shape)
grad_inputs_autograd = torch.autograd.grad(
    [outputs],
    [inputs],
    grad_outputs=[grad_outputs]
)[0]
print("grad_inputs_autograd.shape", grad_inputs_autograd.shape)

'''
summed gradient wrt weight
'''
outputs, _ = layer(inputs)
grad_params = torch.autograd.grad(
    [outputs], layer.parameters(), grad_outputs=[grad_outputs]
)
print("grad_params.shape", [grad_weight.shape for grad_weight in grad_params])

'''
summed gradient wrt weight_ih_l0 with BackPACK
'''
# TODO

'''
individual gradient wrt weight_ih_l0 with autograd
'''
igrad_autograd = [torch.zeros(N, *param.shape) for param in layer.parameters()]
for n in range(N):
    inputs_n = inputs[:, [n]]
    outputs_n, _ = layer(inputs_n)
    grad_outputs_n = grad_outputs[:, [n]]
    grad_weight_n = torch.autograd.grad(
        [outputs_n], layer.parameters(), grad_outputs=[grad_outputs_n]
    )
    for n_param in range(4):
        igrad_autograd[n_param][n] = grad_weight_n[n_param]
print("igrad_autograd.shape", [igrad.shape for igrad in igrad_autograd])

'''
individual gradient wrt weights with BackPACK
'''
layer = extend(layer)
outputs, h_n = layer(inputs)
print("input0", layer.input0.shape)
print("output0", layer.output[0].shape)
print("Is outputs identical to output0?", torch.allclose(outputs, layer.output[0]))
print("Is tanh(h_n) identical to outputs[-1]?",
      torch.allclose(torch.tanh(h_n), outputs[-1]))
print("Is h_n identical to outputs[-1]?",
      torch.allclose(h_n, outputs[-1]))

# compute jacobian based on output0, input0
# jac_h_b = torch.ones(*layer.output[0].shape) - torch.square(layer.output[0])
# print("jacobian backpack", jac_h_b)
jac_h_b_torch = torch.zeros(seq_len, hidden_size, hidden_size)
for i in range(hidden_size):
    for j in range(seq_len):
        grad_outputs_artificial = torch.zeros(*grad_outputs.shape)
        grad_outputs_artificial[j, 0, i] = 1
        jac_h_b_torch[j, :, i] = torch.autograd.grad(
            [outputs[:, 0, :]], layer.parameters(), grad_outputs=[grad_outputs_artificial[:, 0, :]],
            retain_graph=True
        )[2]
# print("jacobian pytorch", jac_h_b_torch)
jac_h_b = torch.zeros(seq_len, hidden_size, hidden_size)
for t in range(seq_len):
    for k in range(hidden_size):
        for i in range(hidden_size):
            fac1 = (1 - outputs[t, 0, k] ** 2)
            fac2 = k == i
            if t > 0:
                for o in range(hidden_size):
                    fac2 += layer.weight_hh_l0[k, o] * jac_h_b[t - 1, i, o]
            jac_h_b[t, i, k] = fac1 * fac2
# print("jacobian backpack", jac_h_b)
print("Jacobian matches pytorch?", torch.allclose(jac_h_b, jac_h_b_torch))
# compute gradient wrt to bias: multiply jacobian with grad_outputs
print("grad_output", grad_outputs)
igrad_bias_ih_pytorch = torch.einsum("thk, tnk -> nh", jac_h_b_torch, grad_outputs)
print("manually computed bias_ih from torch", igrad_bias_ih_pytorch)
print("Does manual and automatic gradient from pytorch match?",
      torch.allclose(igrad_bias_ih_pytorch, igrad_autograd[2]))
igrad_bias_ih_backpack = torch.einsum("thk,tnk -> nh", jac_h_b, grad_outputs)
# compare to pytorch
print("Are bias_ih gradients identical?",
      torch.allclose(igrad_bias_ih_backpack, igrad_autograd[2]))
print("backpack", igrad_bias_ih_backpack)
print("autograd", igrad_autograd[2])
# print(jac_h_b[0,0,0]*grad_outputs[0,0,0]+jac_h_b[1,0,0]*grad_outputs[1,0,0])
print("Are bias gradients identical for input and hidden?",
      torch.allclose(*igrad_autograd[2:4]))
