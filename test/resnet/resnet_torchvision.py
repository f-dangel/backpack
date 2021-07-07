"""Test whether torchvision is extendable with graph utils."""
import torch
import torchvision.models
from torch.nn import MSELoss

from backpack import backpack, extend
from backpack.custom_module.graph_utils import print_table
from backpack.extensions import DiagGGNExact

resnet18 = torchvision.models.resnet18(num_classes=100).eval()
print_table(resnet18)

# convert to backpack
resnet18_transformed = extend(resnet18, use_converter=True)

# print table
print_table(resnet18_transformed)

# compare result
x = torch.randn(8, 3, 21, 21, requires_grad=True)
result_original = resnet18(x)
result_transformed = resnet18_transformed(x)
print(f"\nResults same? {torch.allclose(result_original, result_transformed)}")

# compute second order backward
loss = extend(MSELoss())(result_transformed, torch.rand_like(result_transformed))
with backpack(DiagGGNExact()):
    loss.backward()
for name, param in resnet18_transformed.named_parameters():
    print(name)
    print(param.grad.shape)
    print(param.diag_ggn_exact.shape)
