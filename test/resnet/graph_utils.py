"""Transformation tools to make graph BackPACK compatible."""
from typing import Type

import torch
import torch.fx
from torch.nn import Module

from backpack.custom_module.branching import ActiveIdentity, Branch, Merge
from backpack.custom_module.scale_module import ScaleModule


class MyCustomTracer(torch.fx.Tracer):
    # Inside here you can override various methods
    # to customize tracing. See the `Tracer` API
    # reference
    def is_leaf_module(self, m, module_qualified_name):
        if isinstance(m, ScaleModule):
            return True
        elif isinstance(m, Merge):
            return True
        elif isinstance(m, Branch):
            return True
        elif isinstance(m, ActiveIdentity):
            return True
        else:
            return super().is_leaf_module(m, module_qualified_name)


def transform_mul_to_scale_module(
    module: Module, tracer_class: Type[torch.fx.Tracer]
) -> Module:
    print("\nBegin transformation...")
    graph: torch.fx.Graph = tracer_class().trace(module)
    # FX represents its Graph as an ordered list of
    # nodes, so we can iterate through them.
    for node in graph.nodes:
        if node.op == "call_function":
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
    print("End transformation.\n")
    return torch.fx.GraphModule(module, graph)


def transform_add_to_merge(
    module: Module, tracer_class: Type[torch.fx.Tracer]
) -> Module:
    print("\nBegin transformation...")
    graph: torch.fx.Graph = tracer_class().trace(module)
    # FX represents its Graph as an ordered list of
    # nodes, so we can iterate through them.
    for node in graph.nodes:
        if node.op == "call_function":
            if str(node.target) == "<built-in function add>":
                # TODO: use add_submodule
                name = "merge1"
                setattr(module, name, Merge())
                node.target = name
                # leave node.args
                node.op = "call_module"

    graph.lint()  # Does some checks to make sure the
    # Graph is well-formed.
    # TODO: delete_all_unused_submodules
    print("End transformation.\n")
    return torch.fx.GraphModule(module, graph)
