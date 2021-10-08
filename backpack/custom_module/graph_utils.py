"""Transformation tools to make graph BackPACK compatible."""
from copy import deepcopy
from typing import Tuple, Union
from warnings import warn

from torch.fx import Graph, GraphModule, Node, Tracer
from torch.nn import LSTM, RNN, Dropout, Flatten, Module, Sequential

from backpack.custom_module.branching import SumModule, _Branch
from backpack.custom_module.permute import Permute
from backpack.custom_module.reduce_tuple import ReduceTuple
from backpack.custom_module.scale_module import ScaleModule


class BackpackTracer(Tracer):
    """Tracer that recognizes BackPACK's custom modules as 'leaf modules'."""

    def is_leaf_module(
        self, m: Module, module_qualified_name: str
    ) -> bool:  # noqa: D102
        if isinstance(m, (ScaleModule, SumModule, _Branch, ReduceTuple, Permute)):
            return True
        else:
            return super().is_leaf_module(m, module_qualified_name)


def convert_module_to_backpack(module: Module, debug: bool) -> GraphModule:
    """Convert all modules to BackPACK-compatible modules.

    Transformations:
    - mul -> ScaleModule
    - add -> AddModule
    - flatten -> nn.Flatten
    - getitem -> ReduceTuple
    - permute -> Permute
    - transpose -> Transpose
    - LSTM: split multiple layers
    - inplace -> normal
    - remove duplicates
    - delete unused modules
    - check BackPACK compatible

    Args:
        module: module to convert
        debug: if True prints to command line

    Returns:
        BackPACK-compatible module
    """
    if debug:
        print("\nMake module BackPACK-compatible...")
    module_new = _transform_mul_to_scale_module(module, debug)
    module_new = _transform_flatten_to_module(module_new, debug)
    module_new = _transform_add_to_sum_module(module_new, debug)
    module_new = _transform_get_item_to_module(module_new, debug)
    module_new = _transform_permute_to_module(module_new, debug)
    module_new = _transform_transpose_to_module(module_new, debug)
    module_new = _transform_lstm_rnn(module_new, debug)
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
    """Checks whether the computation graph of the given module is BackPACK compatible.

    More specifically, it checks whether all nodes are either input/output
    or a call to a module. Subsequent checks if the module is extendable in BackPACK
    have to be done by running the extension.

    Args:
        module: module to check
        debug: whether to print debug messages
    """
    if debug:
        print("\tChecking BackPACK compatibility.")
    graph: Graph = BackpackTracer().trace(module)
    for node in graph.nodes:
        if node.op not in ["call_module", "placeholder", "output"]:
            warn(
                f"Encountered node that may break second-order extensions: op={node.op}"
                f", target={node.target}. If you encounter this problem, please open an"
                " issue at https://github.com/f-dangel/backpack/issues."
            )


def _transform_mul_to_scale_module(module: Module, debug: bool) -> GraphModule:
    """Transforms multiplications of tensor with float to ScaleModule.

    Args:
        module: container module to transform
        debug: whether to print debug messages

    Returns:
        equivalent transformed module

    Raises:
        RuntimeError: if a multiplication is found but node.args are not (float, Node)
    """
    target_function = "<built-in function mul>"
    target_method = "multiply"
    if debug:
        print(f"\tBegin transformation: {target_function} -> ScaleModule")

    graph: Graph = BackpackTracer().trace(module)
    nodes_function = [
        n
        for n in graph.nodes
        if n.op == "call_function" and str(n.target) == target_function
    ]
    nodes_method = [
        n
        for n in graph.nodes
        if n.op == "call_method" and str(n.target) == target_method
    ]

    for node in nodes_function:
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
    for node in nodes_method:
        _change_node_to_module(
            node, "scale_module", module, ScaleModule(node.args[1]), (node.args[0],)
        )

    graph.lint()

    if debug:
        print(f"\tMultiplications transformed: {len(nodes_function)+len(nodes_method)}")

    return GraphModule(module, graph)


def _transform_add_to_sum_module(module: Module, debug: bool) -> GraphModule:
    """Transforms summations of tensors to SumModule (useful in ResNets).

    Args:
        module: container module to transform
        debug: whether to print debug messages

    Returns:
        equivalent transformed module
    """
    target_function = "<built-in function add>"
    target_method = "add"
    if debug:
        print(f"\tBegin transformation: {target_function} -> SumModule")

    graph: Graph = BackpackTracer().trace(module)
    nodes = [
        n
        for n in graph.nodes
        if (n.op == "call_function" and str(n.target) == target_function)
        or (n.op == "call_method" and str(n.target) == target_method)
    ]

    for node in nodes:
        _change_node_to_module(node, "sum_module", module, SumModule(), node.args)

    graph.lint()

    if debug:
        print(f"\tSummations transformed: {len(nodes)}")

    return GraphModule(module, graph)


def _transform_flatten_to_module(module: Module, debug: bool) -> GraphModule:
    """Transforms PyTorch's flatten method to the nn.Flatten module.

    Args:
        module: container module to transform
        debug: whether to print debug messages

    Returns:
        equivalent transformed module
    """
    target_function = "<built-in method flatten"
    target_method = "flatten"
    if debug:
        print(f"\tBegin transformation: {target_function} -> Flatten")

    graph: Graph = BackpackTracer().trace(module)
    nodes = [
        n
        for n in graph.nodes
        if (n.op == "call_function" and target_function in str(n.target))
        or (n.op == "call_method" and target_method == str(n.target))
    ]

    for node in nodes:
        start_dim = node.args[1] if len(node.args) > 1 else 0
        end_dim = node.args[2] if len(node.args) > 2 else -1
        _change_node_to_module(
            node, "flatten", module, Flatten(start_dim, end_dim), (node.args[0],)
        )

    graph.lint()

    if debug:
        print(f"\tFlatten functions transformed: {len(nodes)}")

    return GraphModule(module, graph)


def _transform_get_item_to_module(module: Module, debug: bool) -> GraphModule:
    """Transforms the built-in getitem function to ReduceTuple module.

    This function is usually used to reduce the tuple output of RNNs.

    Args:
        module: container module to transform
        debug: whether to print debug messages

    Returns:
        equivalent transformed module
    """
    target = "<built-in function getitem>"
    if debug:
        print(f"\tBegin transformation: {target} -> ReduceTuple")
    graph: Graph = BackpackTracer().trace(module)

    nodes = [
        n for n in graph.nodes if n.op == "call_function" and target == str(n.target)
    ]
    for node in nodes:
        _change_node_to_module(
            node,
            "reduce_tuple",
            module,
            ReduceTuple(index=node.args[1]),
            (node.args[0],),
        )

    graph.lint()
    if debug:
        print(f"\tReduceTuple transformed: {len(nodes)}")
    return GraphModule(module, graph)


def _transform_permute_to_module(module: Module, debug: bool) -> GraphModule:
    """Transforms permute function or method to Permute module.

    Args:
        module: container module to transform
        debug: whether to print debug messages

    Returns:
        equivalent transformed module
    """
    target1 = "permute"
    target2 = "<built-in method permute"
    if debug:
        print(f"\tBegin transformation: {target1}|{target2} -> Permute")
    graph: Graph = BackpackTracer().trace(module)

    nodes = [
        n
        for n in graph.nodes
        if (n.op == "call_function" and target2 in str(n.target))
        or (n.op == "call_method" and target1 == str(n.target))
    ]

    for node in nodes:
        _change_node_to_module(
            node,
            "permute",
            module,
            Permute(*node.args[1]) if len(node.args) == 2 else Permute(*node.args[1:]),
            (node.args[0],),
        )

    graph.lint()
    if debug:
        print(f"\tPermute transformed: {len(nodes)}")
    return GraphModule(module, graph)


def _transform_transpose_to_module(module: Module, debug: bool) -> GraphModule:
    """Transforms transpose function or method to Permute module.

    The Permute module is initialized with transpose parameters and computes
    the permutation on its first forward pass.

    Args:
        module: container module to transform
        debug: whether to print debug messages

    Returns:
        equivalent transformed module
    """
    target_function = "<built-in method transpose"
    target_method = "transpose"
    if debug:
        print(f"\tBegin transformation: {target_method} -> Permute")
    graph: Graph = BackpackTracer().trace(module)

    nodes = [
        n
        for n in graph.nodes
        if (n.op == "call_function" and target_function in str(n.target))
        or (n.op == "call_method" and target_method == str(n.target))
    ]

    for node in nodes:
        _change_node_to_module(
            node,
            "permute",
            module,
            Permute(*node.args[1:], init_transpose=True),
            (node.args[0],),
        )

    graph.lint()
    if debug:
        print(f"\tPermute transformed: {len(nodes)}")
    return GraphModule(module, graph)


def _transform_lstm_rnn(module: Module, debug: bool) -> GraphModule:
    """Transforms multi-layer RNN/LSTM to Sequential of single-layer RNN/LSTM.

    Converts multi-layer RNN/LSTM to Sequential with single-layer RNN/LSTM.
    If dropout probability is nonzero, creates intermediate dropout layers.
    Finally, copies training mode.

    Args:
        module: container module to transform
        debug: whether to print debug messages

    Returns:
        equivalent transformed module

    Raises:
        NotImplementedError: if initial hidden state is used in forward pass
    """
    if debug:
        print("\tBegin transformation: LSTM, RNN")
    graph: Graph = BackpackTracer().trace(module)

    nodes = [
        n
        for n in graph.nodes
        if n.op == "call_module"
        and isinstance(module.get_submodule(n.target), (RNN, LSTM))
        and module.get_submodule(n.target).num_layers > 1
    ]
    for node in nodes:
        if len(node.args) > 1:
            raise NotImplementedError(
                "For conversion, LSTM/RNN input must not have hidden states."
            )
        lstm_module_replace = _make_rnn_backpack(module.get_submodule(node.target))
        module.add_module(node.target, lstm_module_replace)

    graph.lint()
    if debug:
        print(f"\tRNNs, LSTMs transformed: {len(nodes)}")
    return GraphModule(module, graph)


def _rnn_hyperparams(module: Union[RNN, LSTM]) -> Tuple[int, int, float, bool]:
    """Determines the hyperparameters for RNN/LSTM conversion.

    Args:
        module: module to convert

    Returns:
        input_size, hidden_size, dropout, batch_first

    Raises:
        NotImplementedError: if any hyperparameter has a forbidden value
    """
    if module.bias is not True:
        raise NotImplementedError("only bias = True is supported")
    if module.bidirectional is not False:
        raise NotImplementedError("only bidirectional = False is supported")
    if isinstance(module, RNN):
        if module.nonlinearity != "tanh":
            raise NotImplementedError("only nonlinearity = 'tanh' is supported")
    elif isinstance(module, LSTM):
        if module.proj_size != 0:
            raise NotImplementedError("only proj_size = 0 is supported")
    return module.input_size, module.hidden_size, module.dropout, module.batch_first


def _make_rnn_backpack(module: Union[RNN, LSTM]) -> Module:
    """Creates an equivalent module to the multi-layer RNN/LSTM.

    Converts multi-layer RNN/LSTM to Sequential with single-layer RNN/LSTM.
    If dropout probability is nonzero, creates intermediate dropout layers.
    Finally, copies training mode.

    Args:
        module: RNN/LSTM module to convert

    Returns:
        equivalent Sequential module
    """
    input_size, hidden_size, dropout, batch_first = _rnn_hyperparams(module)
    rnn_class = type(module)
    rnn_module_replace = Sequential()
    for layer in range(module.num_layers):
        rnn_layer = rnn_class(
            input_size if layer == 0 else hidden_size,
            hidden_size,
            batch_first=batch_first,
        )
        for param_str in ["weight_ih_l", "weight_hh_l", "bias_ih_l", "bias_hh_l"]:
            setattr(rnn_layer, f"{param_str}0", getattr(module, f"{param_str}{layer}"))
        rnn_module_replace.add_module(f"lstm_{layer}", rnn_layer)
        if layer != (module.num_layers - 1):
            rnn_module_replace.add_module(f"reduce_tuple_{layer}", ReduceTuple())
            if dropout != 0:
                rnn_module_replace.add_module(f"dropout_{layer}", Dropout(dropout))
    rnn_module_replace.train(module.training)
    return rnn_module_replace


def _transform_inplace_to_normal(
    module: Module, debug: bool, initialize_recursion: bool = True
) -> None:
    """Searches for in-place operations and changes them to standard operations.

    Args:
        module: container module to transform
        debug: whether to print debug messages
        initialize_recursion: whether this is the initial call to this function.
    """
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
    """Removes duplicate modules by creating a copy of the module.

    This is necessary because BackPACK saves input/output which is overwritten
    if the module is called multiple times.

    Args:
        module: container module to transform
        debug: whether to print debug messages

    Returns:
        equivalent transformed module

    Raises:
        NotImplementedError: if a duplicate module has parameters
    """
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
    """Find a free name in the modules naming space.

    Args:
        module: the parent module
        initial_name: a name suggestion

    Returns:
        a string with the pattern {initial_name}{int} where module has no such attribute

    Raises:
        RuntimeError: if the module already has an attribute with the intended name
    """

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
