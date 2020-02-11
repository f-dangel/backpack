"""
Einsum utility functions.

Makes it easy to switch to opt_einsum rather than torch's einsum for tests.
"""

import numpy as np
import opt_einsum as oe
import torch

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
    """Use einsum notation for (un-)grouping dimensions.

    Dimensions that cannot be inferred can be handed in via the
    dictionary `dim`.

    Many operations in `backpack` require that certain axes of a tensor
    be treated identically, and will therefore be grouped into a single
    dimesion of the tensor. One way to do that is using `view`s or
    `reshape`s. `eingroup` helps facilitate this process. It can be
    used in the same way as `einsum`, but acts only on a single tensor at
    a time (although this could be fixed with an improved syntax and
    equation analysis).

    Idea:
    -----
    * "a,b,c->ab,c": group dimension a and b into a single one
    * "a,b,c->ba,c" to transpose, then group b and a dimension

    Raises:
    -------
    `KeyError`: If information about a dimension in `dim` is missing
                or can be removed.
    `RuntimeError`: If the groups inferred from `equation` do not match
                    the number of axes of `operand`

    Example usage:
    ```
    import torch
    from backpack.utils.ein import einsum, eingroup

    dim_a, dim_b, dim_c, dim_d = torch.randint(low=1, high=10, size=(4,))
    tensor = torch.randn((dim_a, dim_b, dim_c, dim_d))

    # 1) Transposition: Note the slightly different syntax for `eingroup`
    tensor_trans = einsum("abcd->cbad", tensor)
    tensor_trans_eingroup = eingroup("a,b,c,d->c,b,a,d", tensor)
    assert torch.allclose(tensor_trans, tensor_trans_eingroup)

    # 2) Grouping axes (a,c) and (b,d) together
    tensor_group = einsum("abcd->acbd", tensor).reshape((dim_a * dim_c, dim_b * dim_d))
    tensor_group_eingroup = eingroup("a,b,c,d->ac,bd", tensor)
    assert torch.allclose(tensor_group, tensor_group_eingroup)

    # 3) Ungrouping a tensor whose axes where merged
    tensor_merge = tensor.reshape(dim_a * dim_b, dim_c, dim_d)
    tensor_unmerge = tensor.reshape(dim_a, dim_b, dim_c, dim_d)
    assert torch.allclose(tensor_unmerge, tensor)
    # eingroup needs to know the dimensions of the ungrouped dimension
    tensor_unmerge_eingroup = eingroup(
        "ab,c,d->a,b,c,d", tensor_merge, dim={"a": dim_a, "b": dim_b}
    )
    assert torch.allclose(tensor_unmerge, tensor_unmerge_eingroup)

    # 4) `einsum` functionality to sum out dimensions
    # sum over dim_c, group dim_a and dim_d
    tensor_sum = einsum("abcd->adb", tensor).reshape(dim_a * dim_d, dim_b)
    tensor_sum_eingroup = eingroup("a,b,c,d->ad,b", tensor)
    assert torch.allclose(tensor_sum, tensor_sum_eingroup)
    ```
    """

    dim = {} if dim is None else dim
    in_shape, out_shape, einsum_eq = _eingroup_preprocess(equation, operand, dim=dim)

    operand_in = try_view(operand, in_shape)
    result = einsum(einsum_eq, operand_in)
    return try_view(result, out_shape)


def _eingroup_preprocess(equation, operand, dim):
    """Process `eingroup` equation.

    Return the `reshape`s and `einsum` equations that have to
    be performed.
    """
    split, sep = "->", ","

    def groups(string):
        return string.split(sep)

    lhs, rhs = equation.split(split)
    in_groups, out_groups = groups(lhs), groups(rhs)

    dim = __eingroup_infer(in_groups, operand, dim)
    in_shape_flat, out_shape = __eingroup_shapes(in_groups, out_groups, dim)

    return in_shape_flat, out_shape, equation.replace(sep, "")


def __eingroup_shapes(in_groups, out_groups, dim):
    """Return shape the input needs to be reshaped, and the output shape"""

    def shape(groups, dim):
        return [group_dim(group, dim) for group in groups]

    def group_dim(group, dim):
        try:
            return np.prod([dim[g] for g in group])
        except KeyError as e:
            raise KeyError("Unknown dimension for an axis {}".format(e))

    out_shape = shape(out_groups, dim)

    in_groups_flat = []
    for group in in_groups:
        for letter in group:
            in_groups_flat.append(letter)
    in_shape_flat = shape(in_groups_flat, dim)

    return in_shape_flat, out_shape


def __eingroup_infer(in_groups, operand, dim):
    """Infer the size of each axis."""
    if not len(in_groups) == len(operand.shape):
        raise RuntimeError(
            "Got {} input groups {}, but tensor has {} axes.".format(
                len(in_groups), in_groups, len(operand.shape)
            )
        )

    for group, size in zip(in_groups, operand.shape):
        if len(group) == 1:
            axis = group[0]
            if axis in dim.keys():
                raise KeyError(
                    "Can infer dimension of axis {}.".format(axis),
                    "Remove from dim = {}.".format(dim),
                )
            dim[axis] = size

    return dim


def try_view(tensor, shape):
    """Fall back to reshape (more expensive) if viewing does not work."""
    try:
        return tensor.view(shape)
    except RuntimeError:
        return tensor.reshape(shape)
