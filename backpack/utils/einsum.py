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


def eingroup(equation, operand, dim=None):
    """Use einsum notation for grouping dimensions.

    Dimensions that cannot be inferred can be handed in via the dictionary `dim`.

    Idea:
    -----
    * 'a,b,c->ab,c' to group a and b dimension
    * 'a,b,c->ba,c' to transpose, then group b and a dimension
    """
    dim = {} if dim is None else dim
    in_shape, out_shape, einsum_eq = _eingroup_preprocess(equation, operand, dim=dim)

    operand_in = try_view(operand, in_shape)
    result = einsum(einsum_eq, operand_in)
    return try_view(result, out_shape)


def _eingroup_preprocess(equation, operand, dim):
    split, sep = "->", ","

    def groups(string):
        return string.split(sep)

    def infer(lhs, operand, dim):
        in_groups = groups(lhs)
        assert len(in_groups) == len(operand.shape)

        for group, size in zip(in_groups, operand.shape):
            if len(group) == 1:
                axis = group[0]
                if axis in dim.keys():
                    print("[eingroup] Can infer dim of input group {}".format(group))
                    assert dim[axis] == size
                dim[axis] = size
            else:
                print("[eingroup] Cannot infer dim of input group {}".format(group))

        return dim

    def shape(groups, dim):
        return [group_dim(group, dim) for group in groups]

    def group_dim(group, dim):
        return numpy.prod([dim[g] for g in group])

    lhs, rhs = equation.split(split)
    dim = infer(lhs, operand, dim)

    in_groups, out_groups = groups(lhs), groups(rhs)
    out_shape = shape(out_groups, dim)

    in_groups_flat = []
    for group in in_groups:
        for letter in group:
            in_groups_flat.append(letter)
    in_shape_flat = shape(in_groups_flat, dim)

    return in_shape_flat, out_shape, equation.replace(sep, "")


def try_view(tensor, shape):
    """Fall back to reshape (more expensive) if viewing does not work."""
    try:
        return tensor.view(shape)
    except RuntimeError:
        return tensor.reshape(shape)
