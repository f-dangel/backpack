"""How to use HBP for second-order optimization on a simple model for MNIST.

In this tutorial, we illustrate how to use HBP for training a simple net on
 MNIST.

In particular, we focus on the additional lines that have to be added to an
 already existing training procedure for out-of-the-box PyTorch optimizers.

The model consists of a single linear layer without any activation function.
 For the loss function we choose cross-entropy. The optimizer solves for a
 Newton update with CG.

Outline:
 1. Hyperparameters
 2. Load MNIST
 3. Model
 4. Loss function
 5. Optimizer
 6. Run training procedure

Feel free to play around with the hyperparameters.
"""

# imports
## data loading
import torchvision
import torchvision.transforms as transforms
from os import path

## layers
import torch
from torch.nn import CrossEntropyLoss
from bpexts.hbp.sequential import HBPSequential
from bpexts.hbp.linear import HBPLinear

## for HBP
from bpexts.hbp.loss import batch_summed_hessian

## optimizer
from bpexts.optim.cg_newton import CGNewton

# 1. Hyperparameters
## device: Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
## model
in_features = 784
out_features = 10
bias = True

## training
num_epochs = 2
batch_size = 500

## data directory (directory 'data' in script directory,modify if you like)
data_dir = path.join(path.dirname(path.realpath(__file__)), 'MNIST')

## Curvature matrix: let us choose the Generalized Gauss-Newton matrix
## choices (refer to the paper for more information):
## - 'zero': Generalized Gauss-Newton matrix (GGN)
## - 'abs': Positive-curvature Hessian (PCH) with absolute value casting
## - 'clip': Positive-curvature Hessian (PCH) with value clipping
## - 'none': No treatment of concave terms, NOT useful for optimization
## (Side note: For this example, all modes yield the same curvature,
## but this does not hold for more general networks.
modify_2nd_order_terms = 'zero'

## optimizer
### learning rate
lr = 0.1
### convergence criteria and regularization for CG
cg_maxiter = 50
cg_atol = 0.
cg_tol = 0.1
alpha = 0.02

# 2. Data loading: MNIST
train_set = torchvision.datasets.MNIST(
    root=data_dir, train=True, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True)

# 3. Model
## alternative 1 (used):
linear = HBPLinear(
    in_features=in_features, out_features=out_features, bias=bias)
### HBP has to know how to backward the Hessian. Therefore models must be
### implemented as sequences (from which the order can trivially be inferred)
### HBPSequential is the equivalent to torch.nn.Sequential, but with extended
### Hessian backpropagation functionality
model = HBPSequential(linear).to(device)


## alternative 2 (not used here)
class SimpleModel(HBPSequential):
    """Single linear layer for MNIST.

    Note:
    -----
    HBP has to know how to propagate back the Hessian, hence it is implemented
    as a child of HBPSequential.
    """

    def __init__(self):
        linear = HBPLinear(
            in_features=in_features, out_fearures=out_features, bias=bias)
        super().__init__(linear)


# model = SimpleModel().to(device)

# 4. Loss function
loss_func = CrossEntropyLoss()

# 5. Optimizer
optimizer = CGNewton(
    model.parameters(),
    lr=lr,
    alpha=alpha,
    cg_atol=cg_atol,
    cg_tol=cg_tol,
    cg_maxiter=cg_maxiter)

# 6. Run training procedure
for epoch in range(num_epochs):
    iters = len(train_loader)

    for i, (images, labels) in enumerate(train_loader):
        # reshape and load to device
        images = images.reshape(-1, in_features).to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = loss_func(outputs, labels)

        # set gradients to zero
        optimizer.zero_grad()

        # NOTE: Current implementation only allows for the following order
        ## 1) HBP: Batch average of Hessian of loss w.r.t. model output
        output_hessian = batch_summed_hessian(loss, outputs)
        ## 2) Compute gradients
        loss.backward()
        # 3) HBP: Propagate Hessian back through the graph (note the mode)
        model.backward_hessian(
            output_hessian, modify_2nd_order_terms=modify_2nd_order_terms)

        # 2nd-order optimization step
        optimizer.step()

        # compute and print statistics
        total = labels.size()[0]
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total

        if i % 5 == 0:
            print(
                'Epoch [{}/{}], Iter [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.
                format(epoch + 1, num_epochs, i + 1, iters, loss.item(),
                       accuracy))
