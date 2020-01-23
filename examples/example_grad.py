"""Compute the gradient with PyTorch."""

from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential

from utils import load_mnist_data

B = 4
X, y = load_mnist_data(B)

print("# Gradient with PyTorch | B =", B)

model = Sequential(
    Flatten(),
    Linear(784, 10),
)
lossfunc = CrossEntropyLoss()

loss = lossfunc(model(X), y)
loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(".grad.shape:             ", param.grad.shape)
