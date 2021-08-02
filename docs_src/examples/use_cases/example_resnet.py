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

# %%
# Read-the-docs has limited memory. Therefore, we use a very small problem size.
model = torchvision.models.resnet18(num_classes=5).to(DEVICE)
inputs = rand(4, 3, 7, 7, device=DEVICE)  # (128, 3, 224, 224)

loss_function = extend(MSELoss())

# %%
# Extend resnet18
#
# The network has to be in evaluation mode, because there are BatchNorm layers involved.
# For these, individual gradients can be computed but are not well-defined.
#
# The converter is used for sole reason that in-place functions are not allowed.
# In ResNets, especially x += residual should be x = x + residual instead (torch>=1.9.0).
# If your network does conform to this you don't need to use use_converter=True.
resnet18 = model.eval()
resnet18 = extend(resnet18, use_converter=True)

# %%
# First order extensions work out of the box (apart from in-place functions).
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
# scales with it. In theory, it should work with more classes, but the current implementation
# is not memory efficient enough.
#
# Note: When using the converter the returned module will be a torch.fx.GraphModule.
model = torchvision.models.resnet18(num_classes=5).to(DEVICE)
resnet18 = model.eval()
resnet18 = extend(resnet18, use_converter=True)
print_table(resnet18)

# %%
# Now we can compute second order quantities.
outputs = resnet18(inputs)
loss = loss_function(outputs, rand_like(outputs))
with backpack(DiagGGNExact()):
    loss.backward()

for name, param in resnet18.named_parameters():
    print(name, param.diag_ggn_exact.shape)
