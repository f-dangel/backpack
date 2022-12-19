"""
Compute the gradient with Pytorch
and the variance with BackPACK
"""
from torch.nn import CrossEntropyLoss, Linear
from backpack.utils.examples import load_one_batch_mnist
from backpack import extend, backpack
from backpack.extensions import Variance

X, y = load_one_batch_mnist(flat=True)
model = extend(Linear(784, 10))
lossfunc = extend(CrossEntropyLoss())
loss = lossfunc(model(X), y)

with backpack(Variance()):
    loss.backward()

for param in model.parameters():
    print(param.grad)
    print(param.variance)