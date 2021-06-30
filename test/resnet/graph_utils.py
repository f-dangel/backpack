"""Transformation tools to make graph BackPACK compatible."""
import torch
import torch.fx
from torch.nn import Flatten, Module

from backpack.custom_module.branching import ActiveIdentity, Branch, SumModule
from backpack.custom_module.scale_module import ScaleModule


class MyCustomTracer(torch.fx.Tracer):
    def is_leaf_module(self, m, module_qualified_name):
        if isinstance(m, ScaleModule):
            return True
        elif isinstance(m, SumModule):
            return True
        elif isinstance(m, Branch):
            return True
        elif isinstance(m, ActiveIdentity):
            return True
        else:
            return super().is_leaf_module(m, module_qualified_name)


def transform_mul_to_scale_module(module: Module) -> Module:
    print("\nBegin transformation...")
    graph: torch.fx.Graph = MyCustomTracer().trace(module)
    for node in graph.nodes:
        if node.op == "call_function":
            if str(node.target) == "<built-in function mul>":
                _change_node_to_module(
                    node,
                    "scale_module",
                    module,
                    ScaleModule(node.args[1]),
                    (node.args[0],),
                )

    graph.lint()
    # TODO: delete_all_unused_submodules
    print("End transformation.\n")
    return torch.fx.GraphModule(module, graph)


def transform_add_to_merge(module: Module) -> Module:
    print("\nBegin transformation...")
    graph: torch.fx.Graph = MyCustomTracer().trace(module)
    for node in graph.nodes:
        if node.op == "call_function":
            if str(node.target) == "<built-in function add>":
                _change_node_to_module(node, "merge", module, SumModule(), node.args)

    graph.lint()
    # TODO: delete_all_unused_submodules
    print("End transformation.\n")
    return torch.fx.GraphModule(module, graph)


def transform_flatten_to_module(module: Module) -> Module:
    print("\nBegin transformation...")
    graph: torch.fx.Graph = MyCustomTracer().trace(module)
    for node in graph.nodes:
        if node.op == "call_function":
            if "<built-in method flatten" in str(node.target):
                start_dim = node.args[1] if len(node.args) > 1 else 1
                end_dim = node.args[2] if len(node.args) > 2 else -1
                _change_node_to_module(
                    node,
                    "flatten",
                    module,
                    Flatten(start_dim, end_dim),
                    (node.args[0],),
                )

    graph.lint()
    # TODO: delete_all_unused_submodules
    print("End transformation.\n")
    return torch.fx.GraphModule(module, graph)


def _change_node_to_module(
    node: torch.fx.node.Node,
    name: str,
    base_module: Module,
    new_module: Module,
    args: tuple,
) -> None:
    max_depth = 100
    for i in range(max_depth):
        new_name = f"{name}_{i + 1}"
        if hasattr(base_module, new_name):
            continue
        else:
            break
    name = new_name
    if hasattr(base_module, name):
        raise NotImplementedError(
            f"There already exists a module named {name} in {base_module}. "
            f"Consider increasing max_depth={max_depth} in the module naming space."
        )
    node.op = "call_module"
    node.target = name
    node.args = args
    setattr(base_module, name, new_module)


def print_table(module: Module) -> None:
    graph_resnet18 = MyCustomTracer().trace(module)
    graph_resnet18.print_tabular()
