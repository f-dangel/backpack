r"""Trace Estimation.
=============================

This example illustrates the estimation the Hessian trace of a neural network using Hutchinsons `[Hutchinson, 1990] <https://www.researchgate.net/publication/245083270_A_stochastic_estimator_of_the_trace_of_the_influence_matrix_for_Laplacian_smoothing_splines>`_
method, which is an algorithm to obtain such an an estimate from matrix-vector products:

.. math::
    \text{Let } A \in \mathbb{R}^{D \times D} \text{ be a square matrix and } v \in \mathbb{R}^D
    \text{ be a random vector such that } \mathbb{E}[vv^T] = I. \text{Then,}

.. math::
    \mathrm{Tr}(A) = \mathbb{E}[v^TAv] = \frac{1}{V}\sum_{i=1}^{V}v_i^TAv_i.

A simple derivation for the above can be found in `[Adams, R et al.,] <https://arxiv.org/pdf/1802.03451.pdf>`_.

We will draw v from a Rademacher Distribution and use Hessian-free multiplication to gather matrix-vector products. This can be done with plain autodiff.

Note that there is no dependence between sampled vectors, and the matrix-vector multiplication could in principle be performed in parallel. We can use BackPACK's :code:`HMP` extension to do so, and investigate the potential speedup below.

To verify correctness, we can also obtain the exact trace by summing up the Hessian diagonal elements, computed via :code:`DiagHessian`.
"""


# %%
# Let's get the imports and define the Rademacher distribution

import time

import matplotlib.pyplot as plt
import torch

from backpack import backpack, extend
from backpack.extensions import HMP, DiagHessian
from backpack.hessianfree.hvp import hessian_vector_product
from backpack.utils.examples import load_one_batch_mnist

BATCH_SIZE = 256
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)


def rademacher(shape):
    """Sample from Rademacher distribution."""
    return ((torch.rand(shape) < 0.5)) * 2 - 1


# %%
# Creating the model
# ------------------
#
# We will use a small NN with 2 linear layers without bias. Estimating the trace for a low-dimensional bias of dimension d can simply be done with d Hessian-vector products


model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 20, bias=False),
    torch.nn.Sigmoid(),
    torch.nn.Linear(20, 10, bias=False),
).to(DEVICE)
loss_function = torch.nn.CrossEntropyLoss().to(DEVICE)


# %%
# and we need to ``extend`` the model so that ``BackPACK`` knows about it.
model = extend(model)
loss_function = extend(loss_function)

# %%
# We also use MNIST. Let's load the data and run the forward pass on it. We use ``with(backpack(..))`` syntax to activate two extensions; ``DiagHessian()`` that provides the
# estimation of trace and ``HMP()`` that provides the block Hessian vector product.

x, y = load_one_batch_mnist(BATCH_SIZE)
x, y = x.to(DEVICE), y.to(DEVICE)

loss = loss_function(model(x), y)

with backpack(DiagHessian(), HMP()):
    # keep graph for Hessian-vector products with autodiff
    loss.backward(retain_graph=True)

# %%
# Trace computation
# ----------------------------------------
# Let's compare the Hutchinson trace estimate using 100 samples with the true value.

V = 100

for name, param in model.named_parameters():
    vec = rademacher((V, *param.shape)).to(param.dtype).to(DEVICE)
    Hvec = param.hmp(vec).detach()

    # vᵀHv for each sample 1, ..., V
    vecHvec = torch.einsum(
        "vi,vi->v", vec.flatten(start_dim=1), Hvec.flatten(start_dim=1)
    )
    trace_hutchinson = vecHvec.mean().item()

    trace = param.diag_h.sum().item()

    print("Parameter: ", name)
    print("Trace estimate via Hutchinson, HMP: {:.5f}".format(trace_hutchinson))
    print("Exact trace via DiagHessian       : {:.5f}".format(trace))
    print("*" * 25)

# %%
# We can observe that as the number of samples (V) increases, the approximation of the trace improves. Here is a visualization.

# %%
# Plotting trace approximation accuracy
# -------------------------------------
# Lets get some utility functions first and consider the weights of only one layer to plot it, we can choose any of the linear layers to plot.

name = "1.weight"
for n, p in model.named_parameters():
    if n == name:
        param = p

trace = param.diag_h.sum().item()

V_list = torch.logspace(1, 3, steps=30).int()

fig = plt.figure(figsize=(20, 10))
plt.xlabel("Number of Samples")
plt.ylabel("Trace")

plt.semilogx(V_list, len(V_list) * [trace], color="blue", label="Exact")

num_curves = 15

for i in range(num_curves):
    trace_estimates = []

    for V in V_list:
        torch.manual_seed(i)

        vec = (rademacher((V, *param.shape)).to(param.dtype)).to(DEVICE)
        Hvec = param.hmp(vec).detach()

        # vᵀHv for each sample 1, ..., V
        vecHvec = torch.einsum(
            "vi,vi->v", vec.flatten(start_dim=1), Hvec.flatten(start_dim=1)
        )
        trace_hutchinson = vecHvec.mean().item()

        trace_estimates.append(trace_hutchinson)

    plt.semilogx(
        V_list,
        trace_estimates,
        linestyle="--",
        color="orange",
        label="Hutchinson" if i == 0 else None,
    )

plt.legend()


#%%
# Runtime comparison
# ------------------
# Here, we investigate if the trace estimation can be sped up by vectorizing the Hessian-vector products. We will benchmark the estimation using autodiff Hessian-vector products and block-diagonal vectorized Hessian-vector products from :code:`HMP`.


def hutchinson_trace_autodiff(V):
    """Hessian trace estimate using autodiff Hessian-vector products."""
    trace = 0

    for _ in range(V):
        vec = [rademacher(p.shape).to(p.dtype).to(DEVICE) for p in model.parameters()]
        Hvec = hessian_vector_product(loss, list(model.parameters()), vec)

        for v, Hv in zip(vec, Hvec):
            trace += torch.einsum("i,i->", v.flatten(), Hv.flatten().detach()) / V

    return trace


def hutchinson_trace_hmp(V, V_batch=10):
    """Hessian trace estimate using BackPACK's HMP extension.

    Perform `V_batch` Hessian multiplications at a time.
    """
    V_count = 0
    trace = 0

    while V_count < V:
        missing_V = V - V_count
        next_V = min(V_batch, missing_V)

        for param in model.parameters():
            vec = rademacher((next_V, *param.shape)).to(param.dtype).to(DEVICE)
            Hvec = param.hmp(vec).detach()

            trace += torch.einsum("i,i->", vec.flatten(), Hvec.flatten()) / V

        V_count += next_V

    return trace


# %%
# Let's compare the run time when performing 10 Hessian multiplications in parallel with BackPACK

V = 1000

print("Time for {} samples:".format(V))

V_batch = 10
start = time.time()
trace = hutchinson_trace_hmp(V, V_batch)
end = time.time()
duration = end - start
print("Hutchinson trace with HMP:      {:.5f}, Time {:.5f}".format(trace, duration))

start = time.time()
trace = hutchinson_trace_autodiff(V)
end = time.time()
duration = end - start
print("Hutchinson trace with autodiff: {:.5f}, Time {:.5f}".format(trace, duration))

trace = sum([p.diag_h.sum().item() for p in model.parameters()])
print("(Exact trace: {:.5f})".format(trace))
