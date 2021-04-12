import torch
import time
from backpack import extend

torch.manual_seed(0)

# create fake data
N = 64  # batch size
seq_len = 10
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
start = time.time()
outputs, _ = layer(inputs)
grad_params = torch.autograd.grad(
    [outputs], layer.parameters(), grad_outputs=[grad_outputs]
)
print("grad_params.shape", [grad_weight.shape for grad_weight in grad_params])
end = time.time()
print(f'time for summed gradient torch: {end - start}')

'''
summed gradient wrt weight with BackPACK
'''
# TODO

'''
individual gradient wrt weight with autograd
'''
start = time.time()
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
end = time.time()
print(f'time for complete torch igrad: {end - start}')

'''
individual gradient wrt weights with BackPACK
'''
layer = extend(layer)
outputs, h_n = layer(inputs)
input0 = layer.input0
print("input0", layer.input0.shape)
print("output0", layer.output[0].shape)
# print("Is outputs identical to output0?", torch.allclose(outputs, layer.output[0]))

# compute jacobian based on output0, input0
jac_torch = [
    torch.zeros(seq_len, N, hidden_size, hidden_size, input_size),
    torch.zeros(seq_len, N, hidden_size, hidden_size, hidden_size),
    torch.zeros(seq_len, N, hidden_size, hidden_size),
    torch.zeros(seq_len, N, hidden_size, hidden_size)]
jac_for = [
    torch.zeros(seq_len, N, hidden_size, hidden_size, input_size),
    torch.zeros(seq_len, N, hidden_size, hidden_size, hidden_size),
    torch.zeros(seq_len, N, hidden_size, hidden_size),
    torch.zeros(seq_len, N, hidden_size, hidden_size)]
jac_einsum = [
    torch.zeros(seq_len, N, hidden_size, hidden_size, input_size),
    torch.zeros(seq_len, N, hidden_size, hidden_size, hidden_size),
    torch.zeros(seq_len, N, hidden_size, hidden_size),
    torch.zeros(seq_len, N, hidden_size, hidden_size)]
# extract jacobian from pytorch
for n in range(N):
    for i in range(hidden_size):
        for j in range(seq_len):
            grad_outputs_artificial = torch.zeros(*grad_outputs.shape)
            grad_outputs_artificial[j, n, i] = 1
            gradients_artificial = torch.autograd.grad(
                [outputs[:, n, :]], layer.parameters(), grad_outputs=[grad_outputs_artificial[:, n, :]],
                retain_graph=True
            )
            for n_param in range(4):
                jac_torch[n_param][j, n, :, i] = gradients_artificial[n_param]
# compute jacobian with for loop
start = time.time()
for n in range(N):
    for t in range(seq_len):
        for k in range(hidden_size):
            for i in range(hidden_size):
                fac1 = (1 - outputs[t, n, k] ** 2)
                fac2 = k == i
                if t > 0:
                    for o in range(hidden_size):
                        fac2 += layer.weight_hh_l0[k, o] * jac_for[2][t - 1, n, i, o]
                jac_for[2][t, n, i, k] = fac1 * fac2
end = time.time()
print(f'time for for-loop: {end - start}')
# compute jacobian b_ih and b_hh with einsum
start = time.time()
for t in range(seq_len):
    jac_einsum[2][t, ...] = torch.diag_embed(1 - outputs[t, ...] ** 2, dim1=1, dim2=2)
    if t > 0:
        jac_einsum[2][t, ...] += torch.einsum("nh, hl, nkl -> nkh",
                                              1 - outputs[t, ...] ** 2, layer.weight_hh_l0, jac_einsum[2][t - 1, ...])
end = time.time()
print(f'time for einsum J_h_b: {end - start}')
jac_einsum[3] = jac_einsum[2]
# compute jacobian W_ih with einsum
start = time.time()
for t in range(seq_len):
    jac_einsum[0][t, ...] = torch.einsum("nk, kh, nj -> nkhj",
                                         1 - outputs[t, ...] ** 2, torch.eye(hidden_size), input0[t])
    if t > 0:
        jac_einsum[0][t, ...] += torch.einsum("nh, hl, nklj -> nkhj",
                                              1 - outputs[t, ...] ** 2, layer.weight_hh_l0, jac_einsum[0][t - 1, ...])
end = time.time()
print(f'time for einsum J_h_ih: {end - start}')
# compute jacobian W_hh with einsum
start = time.time()
for t in range(seq_len):
    if t > 0:
        jac_einsum[1][t, ...] = torch.einsum("nk, kh, nj -> nkhj",
                                             1 - outputs[t, ...] ** 2, torch.eye(hidden_size), outputs[t-1])
        jac_einsum[1][t, ...] += torch.einsum("nh, hl, nklj -> nkhj",
                                              1 - outputs[t, ...] ** 2, layer.weight_hh_l0, jac_einsum[1][t-1, ...])
end = time.time()
print(f'time for einsum J_h_hh: {end - start}')

print("All Jacobian J_h_b match?",
      torch.allclose(jac_einsum[2], jac_torch[2]) and
      torch.allclose(jac_einsum[2], jac_for[2]))
print("Jacobian J_h_ih match?",
      torch.allclose(jac_torch[0], jac_einsum[0]))
print("Jacobian J_h_hh match?",
      torch.allclose(jac_torch[1], jac_einsum[1]))

# compute gradient wrt to bias: multiply jacobian with grad_outputs
igrad_bias_ih_pytorch = torch.einsum("tnhk, tnk -> nh", jac_torch[2], grad_outputs)
igrad_bias_ih_backpack = torch.einsum("tnhk,tnk -> nh", jac_for[2], grad_outputs)
igrad_bias_ih_einsum = torch.einsum("tnhk,tnk -> nh", jac_einsum[2], grad_outputs)
print("Are all bias_ih gradients the same?",
      torch.allclose(*igrad_autograd[2:4]) and
      torch.allclose(igrad_bias_ih_einsum, igrad_autograd[2]) and
      torch.allclose(igrad_bias_ih_einsum, igrad_bias_ih_pytorch) and
      torch.allclose(igrad_bias_ih_einsum, igrad_bias_ih_backpack))

igrad_backpack = []
for n_param in range(4):
    if n_param in [0, 1]:
        eq_string = "tnhkj, tnk -> nhj"
    else:
        eq_string = "tnhk, tnk -> nh"
    igrad_backpack.append(torch.einsum(eq_string, jac_einsum[n_param], grad_outputs))
    print(f'Is igrad same for param {n_param}?',
          torch.allclose(igrad_autograd[n_param], igrad_backpack[n_param]))