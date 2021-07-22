"""ResNets in BackPACK
======================
"""

# %%
# Let's get the imports out of the way.
import torch
import torchvision.models
from torch import rand, rand_like
from torch.nn import MSELoss

from backpack import backpack, extend
from backpack.custom_module.graph_utils import print_table
from backpack.extensions import BatchGrad
from backpack.extensions.secondorder.diag_ggn import DiagGGNExact

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

inputs = rand(64, 3, 224, 224, device=DEVICE)
loss_function = extend(MSELoss())


# %%
# Extend resnet18
#
# The network has to be in evaluation mode, because there are BatchNorm layers involved.
# For these, individual gradients can be computed but are not well-defined.
resnet18 = torchvision.models.resnet18(num_classes=10).eval().to(DEVICE)
resnet18 = extend(resnet18)

# %%
# First order extensions work out of the box.
outputs = resnet18(inputs)
loss = loss_function(outputs, rand_like(outputs))
with backpack(BatchGrad()):
    loss.backward()
for name, param in resnet18.named_parameters():
    print(name, param.grad_batch.shape)

# %%
# Second order extensions need to make use of the converter function.
#
# Since second order extensions backpropagate quantities through the whole graph
# all nodes in the computation graph need to have a BackPACK extension.
# This converter function makes sure all nodes are modules that are BackPACK compatible.
# We can verify in the table below that all calls in the graph are modules.
#
# Here, we use a limited number of classes, because the DiagGGN extension memory usage
# scales with it. In theory, should work with more classes, but the current implementation
# is not memory efficient enough.
#
# Note: When using the converter the returned module will be a torch.fx.GraphModule.
resnet18 = torchvision.models.resnet18(num_classes=5).eval().to(DEVICE)
resnet18 = extend(resnet18, use_converter=True)
print_table(resnet18)

# %%
# Now we can compute second order quantities.
#
# and yes, this takes a while (1 min) ...
outputs = resnet18(inputs)
loss = loss_function(outputs, rand_like(outputs))
with backpack(DiagGGNExact()):
    loss.backward()

for name, param in resnet18.named_parameters():
    print(name, param.diag_ggn_exact.shape)
