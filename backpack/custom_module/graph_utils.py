"""Transformation tools to make graph BackPACK compatible."""
from copy import deepcopy
from typing import Set

from torch.fx import Graph, GraphModule, Node, Tracer
from torch.nn import Flatten, Module

from backpack.custom_module.branching import ActiveIdentity, Branch, SumModule
from backpack.custom_module.scale_module import ScaleModule
from backpack.utils import TORCH_VERSION_HIGHER_THAN_1_9_0


class BackPackTracer(Tracer):
    """This custom tracer recognizes BackPACK custom modules."""

    def is_leaf_module(self, m, module_qualified_name):  # noqa: D102
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


def print_table(module: Module) -> None:
    """Prints a table of the module.

    Args:
        module: module to analyze
    """
    graph_resnet18 = BackPackTracer().trace(module)
    graph_resnet18.print_tabular()


def convert_module_to_backpack(module: Module) -> GraphModule:
    """Converts all modules to backpack-compatible modules.

    Transformations:
    - mul -> ScaleModule
    - add -> AddModule
    - flatten -> nn.Flatten

    Args:
        module: module to convert

    Returns:
        backpack compatible module

    Raises:
        NotImplementedError: if not torch >= 1.9.0
    """
    if TORCH_VERSION_HIGHER_THAN_1_9_0 is False:
        raise NotImplementedError(
            "Conversion is only possible for torch >= 1.9.0. This is because these "
            "functions use functionality such as torch.nn.Module.get_submodule"
        )
    print("\nMake module BackPACK compatible...")
    module_new = _transform_mul_to_scale_module(module)
    module_new = _transform_flatten_to_module(module_new)
    module_new = _transform_add_to_sum_module(module_new)
    module_new = _transform_inplace_to_normal(module_new)
    module_new = _transform_remove_duplicates(module_new)
    print("Delete unused modules.")
    module_new.delete_all_unused_submodules()
    print("Finished transformation.\n")
    return module_new


def _transform_mul_to_scale_module(module: Module) -> GraphModule:
    print("Begin transformation: <built-in function mul> -> ScaleModule")
    counter: int = 0
    graph: Graph = BackPackTracer().trace(module)
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
            counter += 1

    graph.lint()
    print(f"Multiplications transformed: {counter}")
    return GraphModule(module, graph)


def _transform_add_to_sum_module(module: Module) -> GraphModule:
    print("Begin transformation: <built-in function add> -> SumModule")
    counter: int = 0
    graph: Graph = BackPackTracer().trace(module)
    for node in graph.nodes:
        if node.op == "call_function":
            if str(node.target) == "<built-in function add>":
                _change_node_to_module(
                    node, "sum_module", module, SumModule(), node.args
                )
                counter += 1

    graph.lint()
    print(f"Summations transformed: {counter}")
    return GraphModule(module, graph)


def _transform_flatten_to_module(module: Module) -> GraphModule:
    print("Begin transformation: <built-in method flatten> -> Flatten")
    counter: int = 0
    graph: Graph = BackPackTracer().trace(module)
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
                counter += 1

    graph.lint()
    print(f"Flatten transformed: {counter}")
    return GraphModule(module, graph)


def _transform_inplace_to_normal(
    module: Module, initialize_recursion: bool = True
) -> Module:
    if initialize_recursion:
        print("Begin transformation: in-place -> standard")
        _transform_inplace_to_normal.counter = 0
    if hasattr(module, "inplace") and module.inplace:
        module.inplace = False
        _transform_inplace_to_normal.counter += 1
    for child_module in module.children():
        _transform_inplace_to_normal(child_module, initialize_recursion=False)

    if initialize_recursion:
        print(f"In-place changed: {_transform_inplace_to_normal.counter}")
        delattr(_transform_inplace_to_normal, "counter")
    return module


def _transform_remove_duplicates(module: Module, max_depth: int = 100) -> GraphModule:
    print("Begin transformation: remove duplicates")
    counter = 0
    graph: Graph = BackPackTracer().trace(module)
    targets: Set[str] = set()
    for node in graph.nodes:
        if node.target in targets:
            original_module = module.get_submodule(node.target)
            for _ in original_module.parameters():
                raise NotImplementedError(
                    f"Transformation not successful, because a cycle was detected. "
                    f"There is a module={original_module} with target={node.target} "
                    f"that is used twice. This detected module has parameters."
                )
            new_module = deepcopy(original_module)
            for i in range(max_depth):
                target = f"{node.target}{i}"
                try:
                    module.get_submodule(target)
                except AttributeError:
                    module.add_submodule(target, new_module)
                    node.target = target
                    counter += 1
                    break
        else:
            targets.add(node.target)

    graph.lint()
    print(f"Duplicates removed: {counter}")
    return GraphModule(module, graph)


def _change_node_to_module(
    node: Node,
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