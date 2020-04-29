---
layout: default
---

# BackPACK on a small example

This small example shows how to use BackPACK to implement a simple second-order optimizer.
It follows [the traditional PyTorch MNIST example](https://github.com/pytorch/examples/tree/master/mnist).


## Installation

For this example to run, you will need [PyTorch and TorchVision (>= 1.0)](https://pytorch.org/get-started/locally/)
To install BackPACK, either use `pip` or [clone the repo](https://github.com/f-dangel/backpack).
```
pip install backpack-for-pytorch
```


## An example: Diagonal GGN Preconditioner

You can find the code 
[in the documentation](https://docs.backpack.pt/en/master/use_cases/example_diag_ggn_optimizer.html).
It runs SGD with a preconditioner based on the diagonal of the GGN.


### Step 1: Libraries, MNIST, and the model

Let's start with the imports and setting some hyperparameters. 
In addition to PyTorch and TorchVision, 
we're going to load the main components we need from BackPACK:

```python
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
```

Now, let's load MNIST


```python

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

```

We'll create a small CNN with MaxPooling and ReLU activations, using a [`Sequential`](https://pytorch.org/docs/stable/nn.html#sequential) layer as the main model.


```python
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

```

We will also need a loss function and a way to measure accuracy

```python
loss_function = torch.nn.CrossEntropyLoss()

def get_accuracy(output, targets):
    """Helper function to print the accuracy"""
    predictions = output.argmax(dim=1, keepdim=True).view_as(targets)
    return predictions.eq(targets).float().mean().item()```

```

### Step 2: The optimizer

The update rule we want to implement is a precondionned gradient descent, 
using the diagonal of the generalized Gauss-Newton,

<center>
<img src="assets/img/updaterule.png" width="60%">
</center>

where `𝛼` is the step-size, `𝜆` is the damping parameter, `g` is the gradient and `G` is the diagonal of the generalized Gauss-Newton (GGN).
The difficult part is computing `G`, but BackPACK will do this;
just like PyTorch's autograd compute the gradient for each parameter `p` and store it in `p.grad`, BackPACK with the `DiagGGNMC` extension will compute (a Monte-Carlo estimate of) the diagonal of the GGN and store it in `p.diag_ggn_mc`.
We can now simply focus on implementing the optimizer that uses this information:

```python
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
```

### Step 3: Put on your BackPACK 

The last thing to do before running the optimizer is (i) tell BackPACK about the model and loss function used and (ii) create the optimizer.

```python
extend(model)
extend(loss_function)

optimizer = DiagGGNOptimizer(
    model.parameters(), 
    step_size=STEP_SIZE, 
    damping=DAMPING
)
```

We are now ready to run!

### The main loop

Traditional optimization loop: load each minibatch, 
compute the minibatch loss, but now call BackPACK before doing the backward pass.
The `diag_ggn_mc` fields of the parameters will get filled and the optimizer will run.

```python
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
```

If everything went fine, the output should look like 

```
Iteration   0/100   Minibatch Loss 2.307   Accuracy 12%
Iteration   1/100   Minibatch Loss 2.318   Accuracy 8%
Iteration   2/100   Minibatch Loss 2.329   Accuracy 8%
Iteration   3/100   Minibatch Loss 2.281   Accuracy 19%
Iteration   4/100   Minibatch Loss 2.265   Accuracy 19%
...
Iteration  96/100   Minibatch Loss 0.319   Accuracy 86%
Iteration  97/100   Minibatch Loss 0.435   Accuracy 89%
Iteration  98/100   Minibatch Loss 0.330   Accuracy 94%
Iteration  99/100   Minibatch Loss 1.227   Accuracy 89%
Iteration 100/100   Minibatch Loss 0.173   Accuracy 95%
```