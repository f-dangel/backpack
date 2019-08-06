"""
Quick example: A small second-order optimizer with BackPACK.

The optimizer implemented here uses a constant damping parameter
and uses the diagonal of the Gauss-Newton/Fisher matrix as a preconditioner;

```
x_{t+1} = x_t - (G_t + bI)^{-1} g_t
```

- `x_t` are the parameters of the model
- `G_t` is the diagonal of the Gauss-Newton/Fisher matrix at `x_t`
- `b` is a damping parameter
- `g_t` is the gradient

"""

from backpack import backpack, extend
from backpack.extensions import DiagGGN
from torch import manual_seed
from torch.nn import CrossEntropyLoss
from torch.optim.optimizer import Optimizer
import example_helper as h


class DiagGGNConstantDampingOptimizer(Optimizer):
    def __init__(self, parameters, damping):
        super().__init__(parameters, dict(damping=damping))

    def step(self, closure=None):
        """
        The closure should reevaluate the model and return the loss.
        """
        with backpack(DiagGGN()):
            loss = closure()
            loss.backward()
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.data.add_(- p.grad / (p.diag_ggn + group["damping"]))
        return loss


# Hyperparameters
BATCH_SIZE = 512
DAMPING = 100.
EPOCHS = 1
manual_seed(0)

# Loading data, creating model and loss function
train_loader = h.mnist_train_loader(batch_size=BATCH_SIZE)
model = h.get_model("2-conv-1-linear")
lossfunc = CrossEntropyLoss()

# Extending the model and loss function using BackPACK
extend(model)
extend(lossfunc)

optimizer = DiagGGNConstantDampingOptimizer(model.parameters(), damping=DAMPING)

if __name__ == "__main__":

    batch_losses = []
    batch_accuracies = []
    batch_grad_norms = []
    batch_variances = []

    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):
            output = model(data)


            def closure():
                return lossfunc(output, target)


            loss = optimizer.step(closure)

            predictions = output.argmax(dim=1, keepdim=True).view_as(target)
            batch_accuracy = predictions.eq(target).float().mean().item()

            h.log(epoch, EPOCHS, batch_idx, train_loader, loss, batch_accuracy)

            batch_accuracies.append(batch_accuracy)
            batch_losses.append(loss.item())

    h.example_2_plot(batch_losses, batch_accuracies)
