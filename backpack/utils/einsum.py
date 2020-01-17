"""
Einsum utility functions.

Makes it easy to switch to opt_einsum rather than torch's einsum for tests.
"""

import opt_einsum as oe
import torch
import numpy

TORCH = "torch"
OPT_EINSUM = "opt_einsum"

BPEXTS_EINSUM = "torch"


def _oe_einsum(equation, *operands):
    # handle old interface, passing operands as one list
    # see https://pytorch.org/docs/stable/_modules/torch/functional.html#einsum
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = operands[0]
    return oe.contract(equation, *operands, backend="torch")


EINSUMS = {
    TORCH: torch.einsum,
    OPT_EINSUM: _oe_einsum,
}


def einsum(equation, *operands):
    """`einsum` implementations used by `backpack`.

    Modify by setting `backpack.utils.utils.BPEXTS_EINSUM`.
    See `backpack.utils.utils.EINSUMS` for supported implementations.
    """
    return EINSUMS[BPEXTS_EINSUM](equation, *operands)


def eingroup(equation, operand):
    """Use einsum notation for grouping dimensions.

    Idea:
    -----
    * 'a,b,c->ab,c' to group a and b dimension
    * 'a,b,c->ba,c' to transpose, then group b and a dimension
    """
    split, sep = "->", ","

    in_string = equation.split(split)[0].replace(sep, "")
    in_shapes = {axis: operand.shape[idx] for idx, axis in enumerate(in_string)}

    out_groups = equation.split(split)[1].split(sep)
    out_shape = []
    for group in out_groups:
        out_shape.append(numpy.prod([in_shapes[axis] for axis in group]))

    einsum_eq = equation.replace(sep, "")
    print(out_shape)
    print(einsum_eq)

    return einsum(einsum_eq, operand).view(out_shape)
