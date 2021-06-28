"""Transformation tools to make graph BackPACK compatible."""
from typing import Type, List

import torch
import torch.fx
from torch.nn import Module

from backpack.custom_module.branching import Merge, Branch, ActiveIdentity
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


def find_branches(module: Module, tracer_class: Type[torch.fx.Tracer], max_depth: int = 100) -> Module:
    print("\nBegin search for double usage of input...")
    list_input: List[torch.fx.node.Node] = []
    list_double_used: List[torch.fx.node.Node] = []
    graph: torch.fx.Graph = tracer_class().trace(module)
    # FX represents its Graph as an ordered list of
    # nodes, so we can iterate through them.
    for node in graph.nodes:
        for argument in node.args:
            assert isinstance(argument, torch.fx.node.Node)
            if argument in list_input:
                if argument not in list_double_used:
                    list_double_used.append(argument)
            else:
                list_input.append(argument)

    for argument in list_double_used:
        print(f"Searching merge point for argument {argument} ...")
        next_nodes: List[torch.fx.node.Node] = []
        for node in graph.nodes:
            if argument in node.args:
                next_nodes.append(node)
        print(f"{argument} has {len(next_nodes)} targets: {next_nodes}.")
        merge_node: torch.fx.node.Node
        for next_node in next_nodes:
            if str(next_node.target) == "<built-in function add>":
                next_nodes.remove(next_node)
                merge_node = next_node
        print(f"\tMerge node: {merge_node}")
        for next_node in next_nodes:
            print(f"\tsearch merge node in other branch starting at {next_node}")
            for _ in range(max_depth):
                if next_node == merge_node:
                    print("\tfound merge node in other branch")
                    break
                next_node = next_node.next

        name = "merge_1"
        setattr(module, name, Merge())
        merge_node.target = name
        # leave merge_node.args
        merge_node.op = "call_module"

        with graph.inserting_after(argument):
            name = "branch_1"
            new_node = graph.call_module(name, args=argument.args)
            setattr(module, name, Branch())
            argument.replace_all_uses_with(new_node)

        merge_node_arg_list = list(merge_node.args)
        for i, arg_node in enumerate(merge_node.args):
            if "branch" in arg_node.target:
                with graph.inserting_after(arg_node):
                    name = "active_identity_1"
                    setattr(module, name, ActiveIdentity())
                    active_identity = graph.call_module(name, args=(arg_node,))
                merge_node_arg_list[i] = active_identity
        merge_node.args = tuple(merge_node_arg_list)

    graph.lint()  # Does some checks to make sure the
    # Graph is well-formed.
    # TODO: delete_all_unused_submodules
    print("End search for double usage.\n")
    return torch.fx.GraphModule(module, graph)

