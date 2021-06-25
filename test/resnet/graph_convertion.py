"""Experiment with the graph convertion tool torch.fx.

Convertion consists of 2 steps:
- converting all <built-in-functions> into CustomModules
- finding parallel structures and replace with Parallel
"""
from typing import Type

import torch.fx
from torch.fx import symbolic_trace
from torch.nn import Module, Sequential, Tanh

from backpack.custom_module.scale_module import ScaleModule


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


def transform_mul_to_scale_module(
    module: Module, tracer_class: Type[torch.fx.Tracer]
) -> Module:
    print("\nBegin transformation...")
    graph: torch.fx.Graph = tracer_class().trace(module)
    # FX represents its Graph as an ordered list of
    # nodes, so we can iterate through them.
    for node in graph.nodes:
        # Checks if we're calling a function (i.e:
        # torch.add)
        if node.op == "call_function":
            # The target attribute is the function
            # that call_function calls.
            if str(node.target) == "<built-in function mul>":
                # TODO: use add_submodule
                name = "scale_module_1"
                setattr(module, name, ScaleModule(node.args[1]))
                node.target = name
                node.args = (node.args[0],)
                node.op = "call_module"

    graph.lint()  # Does some checks to make sure the
    # Graph is well-formed.
    # TODO: delete_all_unused_submodules
    return torch.fx.GraphModule(module, graph)


module_new = transform_mul_to_scale_module(module_original, torch.fx.Tracer)
symbolic_traced_tranformed = symbolic_trace(module_new)
symbolic_traced_tranformed.graph.print_tabular()


class MyCustomTracer(torch.fx.Tracer):
    # Inside here you can override various methods
    # to customize tracing. See the `Tracer` API
    # reference
    def is_leaf_module(self, m, module_qualified_name):
        default = super().is_leaf_module(m, module_qualified_name)
        if default:
            return default
        else:
            return isinstance(m, ScaleModule)


graph_new = MyCustomTracer().trace(module_new)
graph_new.print_tabular()
