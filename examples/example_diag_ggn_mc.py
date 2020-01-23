"""
Compute the gradient with PyTorch and the MC-sampled GGN diagonal with BackPACK.
"""

from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential

from backpack import backpack, extend, extensions
from backpack.utils.examples import load_mnist_data

B = 4
X, y = load_mnist_data(B)

print("# Gradient with PyTorch, MC-sampled GGN diagonal with BackPACK | B =", B)

model = Sequential(Flatten(), Linear(784, 10),)
lossfunc = CrossEntropyLoss()

model = extend(model)
lossfunc = extend(lossfunc)

loss = lossfunc(model(X), y)

with backpack(extensions.DiagGGNMC()):
    loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:       ", param.grad.shape)
    print(".diag_ggn_mc.shape: ", param.diag_ggn_mc.shape)
