"""
Quick example: How to monitor the variance of the minibatch gradients
during training, using BackPACK.
"""

from backpack import backpack, extend
from backpack.extensions import Variance
from torch import manual_seed
from torch.nn import CrossEntropyLoss
from torch.optim.sgd import SGD
import example_helper as h

# Hyperparameters
LR = .1
BATCH_SIZE = 128
EPOCHS = 1
manual_seed(0)

# Loading data, creating model and loss function
train_loader = h.mnist_train_loader(batch_size=BATCH_SIZE)
model = h.get_model("2-conv-1-linear")
lossfunc = CrossEntropyLoss()

# Extending the model and loss function using BackPACK
extend(model)
extend(lossfunc)

optimizer = SGD(model.parameters(), lr=LR)

if __name__ == "__main__":

    batch_losses = []
    batch_accuracies = []
    batch_grad_norms = []
    batch_variances = []

    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):
            output = model(data)
            optimizer.zero_grad()
            loss = lossfunc(output, target)

            with backpack(Variance()):
                loss.backward()
                optimizer.step()

                predictions = output.argmax(dim=1, keepdim=True).view_as(target)
                batch_accuracy = predictions.eq(target).float().mean().item()

                batch_grad_norm = 0
                batch_total_var = 0
                for p in model.parameters():
                    batch_grad_norm += (p.grad ** 2).sum().item()
                    batch_total_var += p.variance.sum().item()
                batch_total_var *= data.shape[0]

            h.log(
                epoch=epoch,
                EPOCHS=EPOCHS,
                batch_idx=batch_idx,
                train_loader=train_loader,
                batch_loss=loss,
                batch_accuracy=batch_accuracy,
                additional="Batch variance: {0:.3f}".format(batch_total_var)
            )

            batch_accuracies.append(batch_accuracy)
            batch_losses.append(loss.item())
            batch_grad_norms.append(batch_grad_norm)
            batch_variances.append(batch_total_var)

    h.example_1_plot(
        batch_losses, batch_accuracies, batch_grad_norms, batch_variances
    )
