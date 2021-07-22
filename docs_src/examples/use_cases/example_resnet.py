"""ResNets in BackPACK
======================
"""
import torch
import torchvision.models
from torch import rand, rand_like
from torch.nn import Flatten, MSELoss

from backpack import backpack, extend
from backpack.custom_module.graph_utils import print_table
from backpack.extensions import BatchGrad
from backpack.extensions.secondorder.diag_ggn import DiagGGNExact
from backpack.utils.examples import load_one_batch_mnist

# %%
# Let's get the imports out of the way.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# Read-the-docs has limited memory. Therefore, we use a smaller custom example.
# Flip the switch to use resnet18.
use_resnet18 = False


class MyFirstResNet(torch.nn.Module):
    def __init__(self, C_in=1, C_hid=5, input_dim=(28, 28), output_dim=10):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(C_in, C_hid, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(C_hid, C_hid, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.flatten = Flatten()
        self.linear1 = torch.nn.Linear(input_dim[0] * input_dim[1] * C_hid, output_dim)
        if C_in == C_hid:
            self.shortcut = torch.nn.Identity()
        else:
            self.shortcut = torch.nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv2(self.relu(self.conv1(x)))
        x += residual
        x = self.flatten(x)
        x = self.linear1(x)
        return x


def get_resnet18():
    model = (
        torchvision.models.resnet18(num_classes=5) if use_resnet18 else MyFirstResNet()
    )
    return model.to(DEVICE)


def get_inputs():
    inputs = (
        rand(64, 3, 224, 224, device=DEVICE)
        if use_resnet18
        else load_one_batch_mnist()[0]
    )
    return inputs.to(DEVICE)


loss_function = extend(MSELoss())

# %%
# Extend resnet18
#
# The network has to be in evaluation mode, because there are BatchNorm layers involved.
# For these, individual gradients can be computed but are not well-defined.
resnet18 = get_resnet18().eval()
resnet18 = extend(resnet18)

# %%
# First order extensions work out of the box.
outputs = resnet18(get_inputs())
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
resnet18 = get_resnet18().eval()
resnet18 = extend(resnet18, use_converter=True)
print_table(resnet18)

# %%
# Now we can compute second order quantities.
outputs = resnet18(get_inputs())
loss = loss_function(outputs, rand_like(outputs))
with backpack(DiagGGNExact()):
    loss.backward()

for name, param in resnet18.named_parameters():
    print(name, param.diag_ggn_exact.shape)
