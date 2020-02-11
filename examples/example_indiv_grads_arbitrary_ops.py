"""Compute the gradient with PyTorch and the variance with BackPACK."""
import torch
from torch.nn import Flatten, Linear, Sequential

from backpack import backpack, extend, extensions

X = torch.randn(size=(50, 784), requires_grad=True)
model = Sequential(Flatten(), extend(Linear(784, 10)),)
loss = torch.mean(torch.sqrt(torch.abs(model(X))))

with backpack(extensions.BatchGrad()):
    loss.backward()

for name, param in model.named_parameters():
    print(name, param.grad_batch.shape)
