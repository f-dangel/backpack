"""
Compute the gradient with Pytorch
and the diagonal of the Gauss-Newton with BackPACK
"""
from torch.nn import CrossEntropyLoss, Linear
from backpack.utils.examples import load_one_batch_mnist
from backpack import extend, backpack
from backpack.extensions import DiagGGNExact

X, y = load_one_batch_mnist(flat=True)
model = extend(Linear(784, 10))
lossfunc = extend(CrossEntropyLoss())
loss = lossfunc(model(X), y)

with backpack(DiagGGNExact()):
    loss.backward()

for param in model.parameters():
    print(param.grad)
    print(param.diag_ggn_exact)
