"""Test whether torchvision is extendable with graph utils."""
from test.resnet.graph_utils import (
    print_table,
    transform_add_to_merge,
    transform_flatten_to_module,
)

import torch
import torchvision.models
from torch.nn import MSELoss

from backpack import backpack, extend
from backpack.extensions import DiagGGNExact

resnet18 = torchvision.models.resnet18()
print_table(resnet18)

# replace add
resnet18_transformed = transform_add_to_merge(resnet18)

# replace flatten
resnet18_transformed = transform_flatten_to_module(resnet18_transformed)

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
