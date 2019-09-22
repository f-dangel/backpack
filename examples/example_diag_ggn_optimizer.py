"""
Quick example: A small second-order optimizer with BackPACK
on the classic MNIST example from PyTorch,
https://github.com/pytorch/examples/blob/master/mnist/main.py

The optimizer we implement uses a constant damping parameter
and uses the diagonal of the GGN/Fisher matrix as a preconditioner;

```
x_{t+1} = x_t - (G_t + bI)^{-1} g_t
```

- `x_t` are the parameters of the model
- `G_t` is the diagonal of the Gauss-Newton/Fisher matrix at `x_t`
- `b` is a damping parameter
- `g_t` is the gradient

"""

import torch
import torchvision
# The main BackPACK functionalities
from backpack import backpack, extend
# The diagonal GGN extension
from backpack.extensions import DiagGGNMC
# This layer did not exist in Pytorch 1.0
from backpack.core.layers import Flatten

# Hyperparameters
BATCH_SIZE = 64
STEP_SIZE = 0.01
DAMPING = 1.0
MAX_ITER = 100
torch.manual_seed(0)


"""
Step 1: Load data and create the model.

We're going to load the MNIST dataset,
and fit a 3-layer MLP with ReLU activations.
"""


mnist_loader = torch.utils.data.dataloader.DataLoader(
    torchvision.datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,)
            )
        ])),
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 20, 5, 1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.Conv2d(20, 50, 5, 1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
    Flatten(), 
    # Pytorch <1.2 doesn't have a Flatten layer
    torch.nn.Linear(4*4*50, 500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 10),
)

loss_function = torch.nn.CrossEntropyLoss()

def get_accuracy(output, targets):
    """Helper function to print the accuracy"""
    predictions = output.argmax(dim=1, keepdim=True).view_as(targets)
    return predictions.eq(targets).float().mean().item()


"""
Step 2: Create the optimizer.

After we call the backward pass with backpack,
every parameter will have a `diag_ggn_mc` field
in addition to a `grad` field.

We can use it to compute the search direction for that parameter,
```
step_direction = p.grad / (p.diag_ggn_mc + group["damping"])
```
and update the weights
"""


class DiagGGNOptimizer(torch.optim.Optimizer):
    def __init__(self, parameters, step_size, damping):
        super().__init__(
            parameters, 
            dict(step_size=step_size, damping=damping)
        )

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                step_direction = p.grad / (p.diag_ggn_mc + group["damping"])
                p.data.add_(-group["step_size"], step_direction)
        return loss



"""
Step 3: Tell BackPACK about the model and loss function, 
create the optimizer, and we will be ready to go
"""

extend(model)
extend(loss_function)

optimizer = DiagGGNOptimizer(
    model.parameters(), 
    step_size=STEP_SIZE, 
    damping=DAMPING
)


"""
Final step: The training loop!

The only difference with a traditional training loop:
Before calling the backward pass, we will call
```
    with backpack(DiagGGNMC()):
```
BackPACK will then add the diagonal of the GGN in the
`diag_ggn_mc` field during the backward pass.
"""


for batch_idx, (x, y) in enumerate(mnist_loader):
    output = model(x)

    accuracy = get_accuracy(output, y)

    with backpack(DiagGGNMC()):
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()

    print(
        "Iteration %3.d/%d   " % (batch_idx, MAX_ITER) +
        "Minibatch Loss %.3f  " % (loss) +
        "Accuracy %.0f" % (accuracy * 100) + "%"
    )

    if batch_idx >= MAX_ITER:
        break
