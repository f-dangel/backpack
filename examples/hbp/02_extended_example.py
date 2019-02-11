"""How to use HBP for second-order optimization of a deep network.

This tutorial is a straightforward extension of the previous one.
However, we will now build a deeper net by combination of different
 layers on use the CIFAR-10 dataset instead of MNIST.

Outline:
 1. Hyperparameters
 2. Load CIFAR-10
 3. Model
 4. Loss function
 5. Optimizer
 6. Run training procedure
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
from bpexts.hbp.relu import HBPReLU
from bpexts.hbp.sigmoid import HBPSigmoid
from bpexts.hbp.combined_relu import HBPReLULinear
from bpexts.hbp.combined_sigmoid import HBPSigmoidLinear

## for HBP
from bpexts.hbp.loss import batch_summed_hessian

## optimizer
from bpexts.optim.cg_newton import CGNewton

# 1. Hyperparameters
## device: Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
## model
in_features = 3072
classes = 10

## training
num_epochs = 10
batch_size = 500

## data directory (directory 'data' in script directory,modify if you like)
data_dir = path.join(path.dirname(path.realpath(__file__)), 'CIFAR10')

## Curvature matrix choices (refer to the paper for more information):
## - 'zero': Generalized Gauss-Newton matrix (GGN)
## - 'abs': Positive-curvature Hessian (PCH) with absolute value casting
## - 'clip': Positive-curvature Hessian (PCH) with value clipping
## - 'none': No treatment of concave terms, NOT useful for optimization
modify_2nd_order_terms = 'abs'

## optimizer
### learning rate
lr = 0.1
### convergence criteria and regularization for CG
cg_maxiter = 50
cg_atol = 0.
cg_tol = 0.1
alpha = 0.05

# 2. Data loading: MNIST
train_set = torchvision.datasets.CIFAR10(
    root=data_dir, train=True, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True)

# 3. Model: Sequence of layers for constructing fully-connected neural nets
## alternative 1 (used):
linear1 = HBPLinear(in_features=in_features, out_features=1024, bias=True)
sigmoid2 = HBPSigmoid()
linear3 = HBPLinear(in_features=1024, out_features=512, bias=False)
sigmoid_linear4 = HBPSigmoidLinear(
    in_features=512, out_features=256, bias=True)
relu_linear5 = HBPReLULinear(in_features=256, out_features=128)
relu6 = HBPReLU()
linear7 = HBPLinear(in_features=128, out_features=classes)

### HBP has to know how to backward the Hessian. Therefore models must be
### implemented as sequences (from which the order can trivially be inferred)
### HBPSequential is the equivalent to torch.nn.Sequential, but with extended
### Hessian backpropagation functionality
model = HBPSequential(linear1, sigmoid2, linear3, sigmoid_linear4,
                      relu_linear5, relu6, linear7).to(device)


## alternative 2 (not used here)
class DeepModel(HBPSequential):
    """Deep model for CIFAR10.

    Note:
    -----
    HBP has to know how to propagate back the Hessian, hence it is implemented
    as a child of HBPSequential.
    """

    def __init__(self):
        linear1 = HBPLinear(
            in_features=in_features, out_features=512, bias=True)
        sigmoid2 = HBPSigmoid()
        linear3 = HBPLinear(in_features=512, out_features=256, bias=False)
        sigmoid_linear4 = HBPSigmoidLinear(
            in_features=256, out_features=128, bias=True)
        relu_linear5 = HBPReLULinear(in_features=128, out_features=64)
        relu6 = HBPRelu()
        linear7 = HPBLinear(in_features=64, out_features=10)
        super().__init__(linear1, sigmoid2, linear3, sigmoid_linear4,
                         relu_linear5, relu6, linear7)


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

        if i % 20 == 0:
            print(
                'Epoch [{}/{}], Iter [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.
                format(epoch + 1, num_epochs, i + 1, iters, loss.item(),
                       accuracy))
