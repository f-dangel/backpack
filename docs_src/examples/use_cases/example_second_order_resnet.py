r"""Second order extensions with a ResNet
========================================

In this example we explain how you can compute second-order information with BackPACK on
models that exhibit branching in the forward pass, one popular class being ResNets.

To tell BackPACK that your model is branched, you will have to use a built-in module
class, instead of writing the forward pass manually.

"""

# %%
# Let's get the imports, configuration and some helper functions out of the way first.

import torch
from torch.nn.utils.convert_parameters import parameters_to_vector

from backpack import backpack, branching, extend
from backpack.extensions import DiagGGNExact
from backpack.hessianfree.ggnvp import ggn_vector_product
from backpack.utils.convert_parameters import vector_to_parameter_list
from backpack.utils.examples import load_one_batch_mnist

BATCH_SIZE = 3
torch.manual_seed(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_accuracy(output, targets):
    """Helper function to print the accuracy"""
    predictions = output.argmax(dim=1, keepdim=True).view_as(targets)
    return predictions.eq(targets).float().mean().item()


x, y = load_one_batch_mnist(batch_size=BATCH_SIZE)
x, y = x.to(DEVICE), y.to(DEVICE)

# %%
# To indicate branching, we build a :py:class:`Parallel <backpack.branching.Parallel>`
# module which feeds its input through all its constituents, then sums up the outputs.
#
# .. note::
#    Similar to :py:class:`torch.nn.Sequential` being a container for a sequence of
#    modules, :py:class:`Parallel <backpack.branching.Parallel>` is the parallel
#    container analogue.


def make_resnet(C_in=1, C_hid=2, input_dim=(28, 28), output_dim=10):
    """Simple ResNet demonstrating the usage of ``Parallel``.

    The forward pass looks as follows

                      ↗—— identity ———↘
    x → conv1 → relu1 → conv2 → relu2 → + → flatten → linear1

    """
    conv1 = torch.nn.Conv2d(C_in, C_hid, kernel_size=3, stride=1, padding=1)
    relu1 = torch.nn.ReLU()
    conv2 = torch.nn.Conv2d(C_hid, C_hid, kernel_size=3, stride=1, padding=1)
    relu2 = torch.nn.ReLU()
    identity = branching.ActiveIdentity()
    flatten = torch.nn.Flatten()
    linear1 = torch.nn.Linear(input_dim[0] * input_dim[1] * C_hid, output_dim)

    return torch.nn.Sequential(
        conv1,
        relu1,
        branching.Parallel(
            identity,
            torch.nn.Sequential(conv2, relu2),
        ),
        flatten,
        linear1,
    )


# %%
# Notice how it is important for second-order extensions to leave the forward pass
# untouched.
#
# .. note:
#    It is currently not possible to use PyTorch's built-in identity module
#    :py:class:`torch.nn.Identity` and instead one has to use BackPACK's
#    :py:class:`ActiveIdentity<backpack.branching.ActiveIdentity>`. The reasons for that
#    are technical: :py:class:`torch.nn.Identity` does not create a new operation in the
#    computation graph. This leads to an unexpected order in which the hooks, that
#    backpropagate information for second-order extensions, are executed.
#
#    The problem is known to the PyTorch developers, and they will resolve this issue
#    in future releases. For now, you need to use
#    :py:class:`ActiveIdentity<backpack.branching.ActiveIdentity>`, which acts like
#    PyTorch's identity, but fixes the backward hook execution order by inserting a new
#    node into the graph during a forward pass.
#
# Let's create and extend our ResNet model, as well as a loss function module:

model = extend(make_resnet()).to(DEVICE)
lossfunc = extend(torch.nn.CrossEntropyLoss(reduction="mean")).to(DEVICE)

# %%
# We can compute the generalized Gauss-Newton/Fisher diagonal as follows:


model.zero_grad()

loss = lossfunc(model(x), y)

with backpack(DiagGGNExact()):
    loss.backward()

print("{:<20}  {:<30} {:<30}".format("Param", "grad", "diag_ggn_exact"))
print("-" * 80)
for name, p in model.named_parameters():
    print(
        "{:<20}: {:<30} {:<30}".format(
            name, str(p.grad.shape), str(p.diag_ggn_exact.shape)
        )
    )

backpack_diag_ggn = [p.diag_ggn_exact for p in model.parameters()]

# %%
# .. note::
#    Currently, BackPACK only supports branching for the second-order extensions
#    :py:class:`DiagGGNExact <backpack.extensions.DiagGGNExact>` and
#    :py:class:`DiagGGNMC<backpack.extensions.DiagGGNMC>`.
#
# To check that everything works, let's compute the GGN diagonal with PyTorch (computing
# one column at a time using GGN-vector products with unit vectors, from which the
# diagonal is cut out).


def autograd_diag_ggn_exact(x, y, model, lossfunc):
    """Compute the generalized Gauss-Newton diagonal via ``autograd``."""
    D = sum(p.numel() for p in model.parameters())
    device = x.device

    outputs = model(x)
    loss = lossfunc(outputs, y)

    ggn_diag = torch.zeros(D, device=device)

    # compute GGN columns by GGNVPs with one-hot vectors
    for d in range(D):
        e_d = torch.zeros(D, device=device)
        e_d[d] = 1.0
        e_d_list = vector_to_parameter_list(e_d, model.parameters())

        ggn_d_list = ggn_vector_product(loss, outputs, model, e_d_list)

        ggn_diag[d] = parameters_to_vector(ggn_d_list)[d]

    return vector_to_parameter_list(ggn_diag, model.parameters())


autograd_diag_ggn = autograd_diag_ggn_exact(x, y, model, lossfunc)

print("Do the GGN diagonals match?")
for ((name, _), backpack_ggn, autograd_ggn) in zip(
    model.named_parameters(), backpack_diag_ggn, autograd_diag_ggn
):
    match = torch.allclose(backpack_ggn, autograd_ggn, atol=1e-7)
    print("{:<20} {}".format(name, match))
