"""
Compute the gradient with PyTorch and the individual gradients' L2 norms with BackPACK.
"""

from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential

from backpack import backpack, extend, extensions
from utils import load_mnist_data

B = 4
X, y = load_mnist_data(B)

print("# Gradient with PyTorch, individual gradients' L2 norms with BackPACK | B =", B)

model = Sequential(
    Flatten(),
    Linear(784, 10),
)
lossfunc = CrossEntropyLoss()

model = extend(model)
lossfunc = extend(lossfunc)

loss = lossfunc(model(X), y)

with backpack(extensions.BatchL2Grad()):
    loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:             ", param.grad.shape)
    print(".batch_l2.shape:         ", param.batch_l2.shape)
