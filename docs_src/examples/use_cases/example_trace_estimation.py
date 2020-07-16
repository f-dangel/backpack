r"""Trace Estimation.
=============================

Hutchinsons `[Hutchinson, 1990] <https://www.researchgate.net/publication/245083270_A_stochastic_estimator_of_the_trace_of_the_influence_matrix_for_Laplacian_smoothing_splines>`_
method is a simple way to obtain an estimate of the trace of a matrix.  

.. math::
    \text{Let } A \in \mathbb{R}^{N \times N} \text{ be a square matrix and } x \in \mathbb{R}^D 
    \text{ be a random vector such that } \mathbb{E}[xx^T] = I. \text{Then,}

.. math::
    Tr(A) = \mathbb{E}[x^TAX] = \frac{1}{N}\sum_{i=1}^{n}x_i^TAx_i

where the random vector x is sampled from a Rademacher Distribution. A simple derivation for the above can be found in `[Adams, R et al.,] <https://arxiv.org/pdf/1802.03451.pdf>`_.

This method is a nice trick, but it is still expensive to calculate the trace of an N x N hessian matrix during backpropagation. Firstly, backPACK provides a function to obtain the
hessian matrix product where we are able to perform 3-D tensor multiplication and this reduces the computation cost quite drastically. Furthermore, backPACK provides a fast and accurate estimation
of the trace of the hessian by obtaining the diagonal and then summing over it! Finally, we are able to obtain the estimation of the trace and we can save enourmous computational cost.  
"""


# %%
# Let's get the imports and define the rademacher distribution

import torch
from torch.optim import Optimizer
import torch.nn as nn

from backpack import backpack, extend
from backpack.hessianfree.hvp import hessian_vector_product
from backpack.extensions import HMP, DiagHessian
from backpack.utils.examples import get_mnist_dataloder
import matplotlib.pyplot as plt
import numpy as np
import time

NUM_EPOCHS = 1
PRINT_EVERY = 50
MAX_ITER = 200
BATCH_SIZE = 512
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
estimated_batch_size = 10 # This parameter is for the loop and plot code found later

def rademacher(shape):
    return ((torch.rand(shape)<0.5))*2-1

# %%
# Creating the model and loading some data
# ----------------------------------------
#
# We will use a small NN with 2 linear layers and avoid bias for now so our dimensions of the weight matrices match.
# We also use MNIST.

def MLP():
    return nn.Sequential(nn.Flatten(), nn.Linear(784,20, bias=False), nn.Sigmoid(), nn.Linear(20,10, bias=False))


mnist_dataloader = get_mnist_dataloder(batch_size=BATCH_SIZE)

model = MLP().to(DEVICE)
loss_function = nn.CrossEntropyLoss().to(DEVICE)

# %%
# and we need to ``extend`` the model so that ``BackPACK`` knows about it.
model = extend(model)
loss_function = extend(loss_function)

#%%
# Here, we load the data and run forward pass on the data. We use ``with(backpack(..))`` syntax to activate two extensions; ``DiagHessian()`` that provides the 
# estimation of trace and ``HMP()`` that provides the hessian matrix product.
x, y = next(iter(mnist_dataloader))
x, y = x.to(DEVICE), y.to(DEVICE)

logits = model(x)
loss = loss_function(logits, y)
with backpack(DiagHessian(), HMP()):
    loss.backward(retain_graph=True)

# %%
# Trace computation
# ----------------------------------------
# Here, we compare the method of trace estimation by hutchinsons and then backPACK. 
# ``Note``: We draw the random vector from Rademacher distribution.

V = 100
for name, param in model.named_parameters():
    vec = rademacher((V, *param.shape)).to(param.dtype).to(DEVICE)

    trace = (param.diag_h).sum()

    trace_hutchinson = (torch.einsum('ijk,ijk->i', vec, param.hmp(vec).detach()).sum()) / V 

    print("Parameter: ", name)
    print("Trace via Hutchinsons: {:.5f} ".format(trace_hutchinson.item()))
    print("Trace via backPACK: {:.5f}".format(trace.item()))

    print("*"*25)

# %%
# We can observe that as the number of samples(V) increases, the approximation of the trace via the hutchinsons method converges to the true trace obtained from the diagonal. To understand this easily, we create a visualization plot below.

# %%
# Plotting trace via backPACK vs Hutchinsons
# ----------------------------------------
# Here, we plot the convergence of the trace approximated via the hutchinsons method to the true trace value obtained from the diagonal. 
# Lets get some util functions first and consider the weights of only one layer to plot it, we can choose any of the linear layers to plot.

parameter = "1.weight"
for n, p in model.named_parameters():
    if n == parameter:
        name, param = n, p


dark = np.array([51.0, 51.0, 51.0]) / 255.0
red = np.array([141.0, 45.0, 57.0]) / 255.0
lred = np.array([1, 1, 1]) - 0.5 * (np.array([1, 1, 1]) - red)
gray = np.array([175.0, 179.0, 183.0]) / 255.0

# %%
# here, we can change the number of samples by tweaking the exponential and samples parameter where exponential decides the scale of the samples.

exponential = 3
samples = 10
sample_list = [pow(samples, i+1) for i in range(exponential)]

# %% The plot is trace (Y-Axis) vs the number of samples (X-Axis). We can observe that as the sample size increases, the trace via hutchinsons method converges to the true trace.
def plotfig(tr, tr_rad):
    plt.xlabel("Number of Samples")
    plt.ylabel("Trace")
    plt.plot(X, tr, color=dark)
    plt.plot(X, tr_rad, color=lred)

X = np.linspace(samples, samples**exponential, exponential)
plt.rcParams["figure.figsize"] = (20, 10)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

tr_hutchinson_all = []
for i in range(8):
    torch.manual_seed(i)
    trace = [(param.diag_h).sum()]*exponential
    tr_hutchinson = []
    for V in sample_list:
        vec = (rademacher((V, *param.shape)).to(param.dtype)).to(DEVICE)
        trace_hutchinson = (torch.einsum('ijk,ijk->i', vec, param.hmp(vec).detach()).sum()).item() / V
        tr_hutchinson.append(trace_hutchinson)
    plotfig(trace, tr_hutchinson)
    tr_hutchinson_all.append(tr_hutchinson)
    
plt.fill_between(X, max(tr_hutchinson_all), min(tr_hutchinson_all), color=gray, alpha=0.3)

#%%
# Runtime comparison
# ----------------------------------------
# Here, we show a quick runtime comparison where we compare the computational cost of the trace of the hessian obtained via a)Hutchinsons method using Autograd b) Hutchinsons method using backPACK c)True trace using backPACK. 

def trace_pytorch(V):
    tr_autodiff = torch.zeros((V, len(list(model.parameters()))))

    for v in range(V):
        vec = [rademacher(p.shape).to(p.dtype).to(DEVICE) for n,p in model.named_parameters()]
        HV = hessian_vector_product(loss, list(model.parameters()), vec)
        trace_autodiff = torch.Tensor([torch.einsum('ij,ij->i', v, hv).sum() for v,hv in zip(vec, HV)])
        tr_autodiff[v] = trace_autodiff 
    trace_autodiff = torch.sum(tr_autodiff, axis=0) / V


def trace_computation_hutchinson(V):
    batch_nos = (V // estimated_batch_size) if V > estimated_batch_size else 1
    tr_hutchinson = torch.zeros((V, len(list(model.parameters()))))
    for _, param in model.named_parameters():
        for i in range(batch_nos):
            vec = (rademacher((estimated_batch_size, *param.shape)).to(param.dtype)).to(DEVICE)
            trace_hutchinson = (torch.einsum('ijk,ijk->i', vec, param.hmp(vec).detach()).sum()).item()
            tr_hutchinson[i] = trace_hutchinson
        trace_hutchinson = torch.sum(tr_hutchinson, axis=0) / V

def trace_computation_backpack():
    for _, param in model.named_parameters():
        trace = (param.diag_h).sum()

# %%        
# we can change V and test the time accordingly.

V = 10000
print("Time taken for {} vectors:".format(V))
start_hut = time.time()
trace_computation_hutchinson(V)
print("trace via hutchinsons with backPACK: {:.5f}".format(time.time() - start_hut))

start_bp = time.time()
trace_computation_backpack()
print("true trace with backPACK: {:.5f}".format(time.time() - start_bp))

start_ad = time.time()
trace_pytorch(V)
print("trace via hutchinsons with Autodiff: {:.5f}".format(time.time() - start_ad))


