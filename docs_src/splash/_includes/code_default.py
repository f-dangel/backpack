"""
Compute the gradient with Pytorch

"""
from torch.nn import CrossEntropyLoss, Linear
from backpack.utils.examples import load_one_batch_mnist


X, y = load_one_batch_mnist(flat=True)
model = Linear(784, 10)
lossfunc = CrossEntropyLoss()
loss = lossfunc(model(X), y)


loss.backward()

for param in model.parameters():
    print(param.grad)
