"""Test whether torchvision is extendable with graph utils."""
import torchvision.models

from test.resnet.graph_utils import MyCustomTracer

resnet18 = torchvision.models.resnet18()
graph_resnet18 = MyCustomTracer().trace(resnet18)
graph_resnet18.print_tabular()

# replace add

# replace flatten

# compare result

# compute second order backward
