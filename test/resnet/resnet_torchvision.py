"""Test whether torchvision is extendable with graph utils."""
import torch
import torchvision.models
from torch.nn import MSELoss

from backpack import backpack, extend
from backpack.custom_module.graph_utils import convert_module_to_backpack, print_table
from backpack.extensions import DiagGGNExact

resnet18 = torchvision.models.resnet18()
print_table(resnet18)

# convert to backpack
resnet18_transformed = convert_module_to_backpack(resnet18)

# print table
print_table(resnet18_transformed)

# compare result
x = torch.randn(128, 3, 21, 21, requires_grad=True)
result_original = resnet18(x)
resnet18_extend = extend(resnet18_transformed, is_container=True)
result_transformed = resnet18_extend(x)
print(f"\nResults same? {torch.allclose(result_original, result_transformed)}")

# compute second order backward
loss = extend(MSELoss())(result_transformed, torch.rand_like(result_transformed))
with backpack(DiagGGNExact()):
    loss.backward()
