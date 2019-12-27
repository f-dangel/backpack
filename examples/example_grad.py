"""Compute the gradient with PyTorch."""

from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential

from utils import load_mnist_data

B = 64
X, y = load_mnist_data(B)

print(f"# COMPUTE THE GRADIENT WITH PYTORCH (B={B})")

model = Sequential(
    Flatten(),
    Linear(784, 10),
)
lossfunc = CrossEntropyLoss()

loss = lossfunc(model(X), y)
loss.backward()

for name, param in model.named_parameters():
    print(
        f"\n{name}:",
        f"\n\t.grad.shape: {param.grad.shape}",
        f"\n\t.grad:       {param.grad}",
    )
