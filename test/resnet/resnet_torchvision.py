"""Test whether torchvision is extendable with graph utils."""
from test.resnet.graph_utils import (
    print_table,
    transform_add_to_merge,
    transform_flatten_to_module,
)

import torch
import torchvision.models

resnet18 = torchvision.models.resnet18()
print_table(resnet18)

# replace add
resnet18_transformed = transform_add_to_merge(resnet18)

# replace flatten
resnet18_transformed = transform_flatten_to_module(resnet18_transformed)

# print table
print_table(resnet18_transformed)

# compare result
x = torch.randn(128, 3, 21, 21)
result_original = resnet18(x)
result_transformed = resnet18_transformed(x)
print(f"\nResults same? {torch.allclose(result_original, result_transformed)}")

# compute second order backward
