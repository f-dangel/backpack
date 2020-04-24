"""
Compute the gradient with PyTorch and the gradient variance with BackPACK.
"""

from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential

from backpack import backpack, extend, extensions
from backpack.utils.examples import load_one_batch_mnist

B = 4
X, y = load_one_batch_mnist(B)

print("# Gradient with PyTorch, gradient variance with BackPACK | B =", B)

model = Sequential(Flatten(), Linear(784, 10),)
lossfunc = CrossEntropyLoss()

model = extend(model)
lossfunc = extend(lossfunc)

loss = lossfunc(model(X), y)

with backpack(extensions.Variance()):
    loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print(".variance.shape:         ", param.variance.shape)
