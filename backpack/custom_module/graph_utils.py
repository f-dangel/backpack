"""Transformation tools to make graph BackPACK compatible."""
from copy import deepcopy
from warnings import warn

from torch.fx import Graph, GraphModule, Node, Tracer
from torch.nn import Flatten, Module

from backpack.custom_module.branching import ActiveIdentity, SumModule, _Branch
from backpack.custom_module.scale_module import ScaleModule
from backpack.utils import TORCH_VERSION_AT_LEAST_1_9_0


class BackpackTracer(Tracer):
    """Tracer that recognizes BackPACK's custom modules as 'leaf modules'."""

    def is_leaf_module(
        self, m: Module, module_qualified_name: str
    ) -> bool:  # noqa: D102
        if isinstance(m, (ScaleModule, SumModule, _Branch, ActiveIdentity)):
            return True
        else:
            return super().is_leaf_module(m, module_qualified_name)


def convert_module_to_backpack(module: Module, debug: bool) -> GraphModule:
    """Convert all modules to BackPACK-compatible modules.

    Transformations:
    - mul -> ScaleModule
    - add -> AddModule
    - flatten -> nn.Flatten
    - inplace to normal
    - remove duplicates
    - delete unused modules
    - check BackPACK compatible

    Args:
        module: module to convert
        debug: if True prints to command line

    Returns:
        BackPACK-compatible module

    Raises:
        NotImplementedError: if not torch >= 1.9.0
    """
    if TORCH_VERSION_AT_LEAST_1_9_0 is False:
        raise NotImplementedError(
            "Conversion is only possible for torch >= 1.9.0. This is because these "
            "functions use functionality such as torch.nn.Module.get_submodule"
        )
    if debug:
        print("\nMake module BackPACK-compatible...")
    module_new = _transform_mul_to_scale_module(module, debug)
    module_new = _transform_flatten_to_module(module_new, debug)
    module_new = _transform_add_to_sum_module(module_new, debug)
    _transform_inplace_to_normal(module_new, debug)
    module_new = _transform_remove_duplicates(module_new, debug)
    if debug:
        print("\tDelete unused modules.")
    module_new.delete_all_unused_submodules()
    _check_backpack_compatible(module_new, debug)
    if debug:
        print("Finished transformation.\n")
    return module_new


def _check_backpack_compatible(module: Module, debug: bool) -> None:
    if debug:
        print("\tChecking BackPACK compatibility.")
    graph: Graph = BackpackTracer().trace(module)
    for node in graph.nodes:
        if not (node.op in ["call_module", "placeholder", "output"]):
            warn(
                f"While checking for BackPACK compatibility, found a node in the "
                f"computation graph that might not be BackPACK compatible. "
                f"The node found has op={node.op}, target={node.target}."
                f"This is an issue in computing second order quantities. "
                f"Please file an issue on the BackPACK repository."
            )


def _transform_mul_to_scale_module(module: Module, debug: bool) -> GraphModule:
    target = "<built-in function mul>"
    if debug:
        print(f"\tBegin transformation: {target} -> ScaleModule")

    graph: Graph = BackpackTracer().trace(module)
    nodes = [
        n for n in graph.nodes if n.op == "call_function" and str(n.target) == target
    ]

    for node in nodes:
        if len(node.args) != 2:
            raise RuntimeError(f"Expecting 2 arguments, got {len(node.args)}.")

        idx_weight = 0 if isinstance(node.args[0], float) else 1
        idx_tensor = 1 - idx_weight

        weight = node.args[idx_weight]
        tensor = node.args[idx_tensor]

        if not (isinstance(weight, float) and isinstance(tensor, Node)):
            raise RuntimeError(
                f"Expecting types [float, Node], got {[type(weight), type(tensor)]}."
            )

        _change_node_to_module(
            node, "scale_module", module, ScaleModule(weight), (tensor,)
        )

    graph.lint()

    if debug:
        print(f"\tMultiplications transformed: {len(nodes)}")

    return GraphModule(module, graph)


def _transform_add_to_sum_module(module: Module, debug: bool) -> GraphModule:
    target = "<built-in function add>"
    if debug:
        print(f"\tBegin transformation: {target} -> SumModule")

    graph: Graph = BackpackTracer().trace(module)
    nodes = [
        n for n in graph.nodes if n.op == "call_function" and str(n.target) == target
    ]

    for node in nodes:
        _change_node_to_module(node, "sum_module", module, SumModule(), node.args)

    graph.lint()

    if debug:
        print(f"\tSummations transformed: {len(nodes)}")

    return GraphModule(module, graph)


def _transform_flatten_to_module(module: Module, debug: bool) -> GraphModule:
    target = "<built-in method flatten"
    if debug:
        print(f"\tBegin transformation: {target} -> Flatten")

    graph: Graph = BackpackTracer().trace(module)
    nodes = [
        n for n in graph.nodes if n.op == "call_function" and target in str(n.target)
    ]

    for node in nodes:
        start_dim = node.args[1] if len(node.args) > 1 else 1
        end_dim = node.args[2] if len(node.args) > 2 else -1
        _change_node_to_module(
            node, "flatten", module, Flatten(start_dim, end_dim), (node.args[0],)
        )

    graph.lint()

    if debug:
        print(f"\tFlatten transformed: {len(nodes)}")

    return GraphModule(module, graph)


def _transform_inplace_to_normal(
    module: Module, debug: bool, initialize_recursion: bool = True
) -> None:
    if initialize_recursion:
        if debug:
            print("\tBegin transformation: in-place -> standard")
        _transform_inplace_to_normal.counter = 0
    if hasattr(module, "inplace") and module.inplace:
        module.inplace = False
        _transform_inplace_to_normal.counter += 1
    for child_module in module.children():
        _transform_inplace_to_normal(child_module, debug, initialize_recursion=False)

    if initialize_recursion:
        if debug:
            print(f"\tIn-place changed: {_transform_inplace_to_normal.counter}")
        del _transform_inplace_to_normal.counter


def _transform_remove_duplicates(module: GraphModule, debug: bool) -> GraphModule:
    if debug:
        print("\tBegin transformation: remove duplicates")

    graph: Graph = BackpackTracer().trace(module)

    targets = [n.target for n in graph.nodes]
    duplicates = {t for t in targets if targets.count(t) > 1}
    nodes = [n for n in graph.nodes if n.target in duplicates]

    for node in nodes:
        target = node.target
        original_module = module.get_submodule(target)

        for _ in original_module.parameters():
            raise NotImplementedError(
                f"Cycle with parameters detected: module {original_module} with target"
                f" {target} has parameters and is used {targets.count(target)} times."
            )

        new_module = deepcopy(original_module)
        new_target = _get_free_name(module, target)
        module.add_submodule(new_target, new_module)
        node.target = new_target

    graph.lint()

    if debug:
        print(f"\tDuplicates removed: {len(nodes)}")

    return GraphModule(module, graph)


def _change_node_to_module(
    node: Node,
    name: str,
    base_module: Module,
    new_module: Module,
    args: tuple,
) -> None:
    """Helper function to change an existing node to a module.

    The new module is registered in the base_module as a submodule.
    The attribute name is based on name{int}.
    The attributes of the node are changed so they point onto the new module.

    Args:
        node: existing node
        name: proposed name, real name is name{int}
        base_module: the module that should get new_module as a child
        new_module: the new module to register on the node and base_module
        args: arguments of the new node
    """
    new_name = _get_free_name(base_module, name)
    node.op = "call_module"
    node.target = new_name
    node.args = args
    setattr(base_module, new_name, new_module)


def _get_free_name(module: Module, initial_name: str) -> str:
    def _has_target(target: str) -> bool:
        try:
            module.get_submodule(target)
            return True
        except AttributeError:
            return False

    counter = 0
    while _has_target(f"{initial_name}{counter}"):
        counter += 1
    name = f"{initial_name}{counter}"

    if hasattr(module, name):
        raise RuntimeError(
            f"Unable to find a free name for registering a new module."
            f"module={module} already has an attribute named {name}."
        )

    return name
