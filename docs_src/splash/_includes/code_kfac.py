"""
Compute the gradient with Pytorch
and KFAC with BackPACK
"""
from torch.nn import CrossEntropyLoss, Linear
from backpack.utils.examples import load_one_batch_mnist
from backpack import extend, backpack
from backpack.extensions import KFAC

X, y = load_one_batch_mnist(flat=True)
model = extend(Linear(784, 10))
lossfunc = extend(CrossEntropyLoss())
loss = lossfunc(model(X), y)

with backpack(KFAC()):
    loss.backward()

for param in model.parameters():
    print(param.grad)
    print(param.kfac)