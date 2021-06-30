"""Experiment with the graph convertion tool torch.fx.

Convertion consists of 2 steps:
- converting all <built-in-functions> into CustomModules
- finding parallel structures and replace with Parallel
"""
from test.resnet.graph_utils import (
    print_table,
    transform_add_to_merge,
    transform_mul_to_scale_module,
)

import torch.fx
from torch.fx import symbolic_trace
from torch.nn import Module, MSELoss, Sequential, Tanh

from backpack import backpack, extend
from backpack.extensions import DiagGGNExact


class ModuleOriginal(Module):
    def __init__(self, in_dim: int, hidden_dim: int, dt: float):
        super(ModuleOriginal, self).__init__()
        self.lin1 = torch.nn.Linear(in_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = torch.nn.Linear(hidden_dim, in_dim)
        self.activation_function_1 = Tanh()
        self.activation_function_2 = Tanh()
        self.net = Sequential(
            self.lin1,
            self.activation_function_1,
            self.lin2,
            self.activation_function_2,
            self.lin3,
        )
        self.dt = dt

    def forward(self, input):
        return input + self.net(input) * self.dt


module_original = ModuleOriginal(2, 10, 0.1)

symbolic_traced: torch.fx.GraphModule = symbolic_trace(module_original)

# High-level intermediate representation (IR) - Graph representation
print(symbolic_traced.graph)

# Code generation - valid Python code
print(symbolic_traced.code)

# print table of graph
symbolic_traced.graph.print_tabular()

# replace all multiplication operations with ScaleModule
module_new = transform_mul_to_scale_module(module_original)
print_table(module_new)

# find branching
module_new = transform_add_to_merge(module_new)
print_table(module_new)

# extend
module_new = extend(module_new, is_container=True)
loss_function = extend(MSELoss())

# define input and solution
x = torch.tensor([[1.0, 2.0]], requires_grad=True)
solution = torch.tensor([[1.0, 1.0]])

logits_original = module_original(x)
logits_new = module_new(x)
print("\nAre the nets equivalent?", torch.allclose(logits_original, logits_new), "\n")

loss_new = loss_function(logits_new, solution)
with backpack(DiagGGNExact()):
    loss_new.backward()

for name, param in module_new.named_parameters():
    print(name)
    print(param.diag_ggn_exact.shape)
