from torch import no_grad
import torch
from torch.nn import Linear, CrossEntropyLoss, Sequential
from torch.optim.sgd import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

import matplotlib.pyplot as plt
from backpack import extend, backpack, extensions as extensions
from backpack.core.layers import Flatten

PROFILE = False

UPDATE_INTERVAL = 10
DATA_FOLDER = '../data'
BATCH_SIZE = 64
SEED = 0
EPOCHS = 1
LR = 0.005

torch.manual_seed(SEED)


def get_mnist_train_loader():
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_dataset = datasets.MNIST(DATA_FOLDER, train=True, download=True, transform=mnist_transform)
    return DataLoader(mnist_dataset, batch_size=BATCH_SIZE, shuffle=True)

def make_plot(iter_ids, losses, norms, variances):
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(iter_ids, losses)

    ax2.plot(iter_ids, norms, label="Gradient norm")
    ax2.plot(iter_ids, variances, label="Gradient variance")

    ax1.set_title("Loss over time")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Cross-entropy")

    ax2.set_title("Gradient norm and variance over time")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Gradient squared")
    ax2.legend()

    plt.show()


if __name__ == "__main__":

    train_loader = get_mnist_train_loader()
    model = Sequential(extend(Flatten()), extend(Linear(784, 10)))
    lossfunc = extend(CrossEntropyLoss())
    optimizer = SGD(params=model.parameters(), lr=LR)

    iter_ids = []
    losses = []
    norms = []
    variances = []

    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):

            optimizer.zero_grad()
            with backpack(extensions.VARIANCE):
                loss = lossfunc(model(data), target)
                loss.backward()

                optimizer.step()

                if batch_idx % UPDATE_INTERVAL == 0:
                    grad_norm = sum([torch.norm(p.grad) ** 2 for p in model.parameters()])
                    total_noise = sum([torch.sum(p.variance) for p in model.parameters()]) * BATCH_SIZE

                    print(
                        "Epoch {}/{}, ".format(epoch, EPOCHS) +
                        "iter {}/{} : ".format(batch_idx, len(train_loader)) +
                        "Loss: {:.3f} ".format(loss) +
                        "Gradient norm: {:.4f} ".format(grad_norm) +
                        "Gradient variance: {:.4f}".format(total_noise)
                    )

                    iter_ids.append(batch_idx)
                    losses.append(loss)
                    norms.append(grad_norm)
                    variances.append(total_noise)

    make_plot(iter_ids, losses, norms, variances)
