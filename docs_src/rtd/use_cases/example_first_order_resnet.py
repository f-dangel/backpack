r"""First order extensions with a ResNet
========================================

"""

# %%
# Let's get the imports, configuration and some helper functions out of the way first.

import torch

from backpack import backpack, extend
from backpack.extensions import BatchGrad
from backpack.utils.examples import load_one_batch_mnist
import torch.nn.functional as F

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
#
#


class MyFirstResNet(torch.nn.Module):
    def __init__(self, C_in=1, C_hid=5, input_dim=(28, 28), output_dim=10):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(C_in, C_hid, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(C_hid, C_hid, kernel_size=3, stride=1, padding=1)
        self.linear1 = torch.nn.Linear(input_dim[0] * input_dim[1] * C_hid, output_dim)
        if C_in == C_hid:
            self.shortcut = torch.nn.Identity()
        else:
            self.shortcut = torch.nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv2(F.relu(self.conv1(x)))
        x += residual
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x


model = extend(MyFirstResNet()).to(DEVICE)

# %%
#
#

model.zero_grad()
loss = F.cross_entropy(model(x), y, reduction="sum")
with backpack(BatchGrad()):
    loss.backward()

print("{:<20}  {:<30} {:<30}".format("Param", "grad", "grad (batch)"))
print("-" * 80)
for name, p in model.named_parameters():
    print(
        "{:<20}: {:<30} {:<30}".format(name, str(p.grad.shape), str(p.grad_batch.shape))
    )

# %%
#
#

sample_to_check = 1
x_to_check = x[sample_to_check, :].unsqueeze(0)
y_to_check = y[sample_to_check].unsqueeze(0)

model.zero_grad()
loss = F.cross_entropy(model(x_to_check), y_to_check)
loss.backward()

print("Do the individual gradient match?")
for param_id, (name, p) in enumerate(model.named_parameters()):
    print(
        name, torch.allclose(p.grad_batch[sample_to_check, :], p.grad, atol=1e-7),
    )
