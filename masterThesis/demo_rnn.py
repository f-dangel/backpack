import torch
torch.manual_seed(0)

# create fake data
N = 128     # batch size
seq_len = 10
input_size = 5
hidden_size = 3
inputs = torch.rand(seq_len, N, input_size)
inputs.requires_grad = True
print("inputs.shape", inputs.shape)

### backpropagate with torch.autograd ###
layer = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1)
outputs, _ = layer(inputs)
print("outputs.shape", outputs.shape)
grad_outputs = torch.rand_like(outputs)
grad_inputs_autograd = torch.autograd.grad(
    [outputs],
    [inputs],
    grad_outputs=[grad_outputs]
)[0]
print("grad_inputs_autograd.shape", grad_inputs_autograd.shape)

### summed gradient wrt weight_ih_l0 ###
outputs, _ = layer(inputs)
torch.autograd.backward(outputs, grad_outputs, retain_graph=True)
grad_weight_backward = layer.weight_ih_l0.grad
grad_weight = torch.autograd.grad(
    [outputs], [layer.weight_ih_l0], grad_outputs=[grad_outputs]
)[0]
print("grad_weight.shape", grad_weight.shape)
print("Does backward() pass do the same as grad()?",
      torch.allclose(grad_weight_backward, grad_weight))

### summed gradient wrt weight_ih_l0 with BackPACK ###
# TODO

### individual gradient wrt weight_ih_l0 with autograd ###
igrad_autograd = torch.zeros(N, *layer.weight_ih_l0.shape)
for n in range(N):
    inputs_n = inputs[:, [n]]
    outputs_n, _ = layer(inputs_n)
    grad_outputs_n = grad_outputs[:, [n]]
    grad_weight_n = torch.autograd.grad(
        [outputs_n], [layer.weight_ih_l0], grad_outputs=[grad_outputs_n]
    )[0]
    igrad_autograd[n] = grad_weight_n
print("igrad_autograd.shape", igrad_autograd.shape)

### individual gradient wrt weight_ih_l0 with BackPACK ###
# TODO
