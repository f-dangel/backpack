""" Einsum utility functions. """

import torch


def eingroup(equation, operand, dim=None):
    """Use einsum-like notation for (un-)grouping dimensions.

    Dimensions that cannot be inferred can be handed in via a mapping `dim`.

    Arguments:
        equation (str): Equation specifying the (un-)grouping of axes.
        operand (torch.Tensor): The tensor that `equation` will be applied to.
        dim (dict, optional): A mapping from letters in `equation` to
            dimensions. Only required if `eingroup` cannot infer the dimension.
            For instance, consider you want to interpret a vector with 10
            elements as a 5x2 matrix. The equation `"i,j->ij"` is not
            sufficient, you need to specify `dim = {"i": 5, "j": 2}`.

    Note:
        Many operations in `backpack` require that certain axes of a tensor
        be treated identically, and will therefore be grouped into a single
        dimension. One way to do that is using `view`s or `reshape`s.
        `eingroup` helps facilitate this process. It can be used in roughly
        the same way as `einsum`, but acts only on a single tensor at
        a time (although this could be fixed with an improved syntax and
        equation analysis).

        Idea:
        * `"a,b,c->ab,c"`: group dimension `a` and `b` into a single one.
        * `"a,b,c->ba,c"` to transpose, then group `b` and `a` dimension.

    Examples:
        Different reshapes of a [2 x 2 x 2] tensor:
        >>> t = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7]).reshape(2, 2, 2)
        >>> t_flat = t.reshape(-1)
        >>> # group all dimensions
        >>> eingroup("i,j,k->ijk", t)
        torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7])
        >>> # interpret as 2 x 4 matrix
        >>> eingroup("i,j,k->i,jk", t)
        torch.Tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        >>> # grouping (specifying grouping dimensions)
        >>> eingroup("ijk->i,j,k", dim={"i": 2, "j": 2, "k": 2})
        torch.Tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
        >>> # grouping with additional contraction
        >>> eingroup("ijk->j,k", dim={"j": 2, "k": 2})
        torch.Tensor([[[4, 5], [8, 10]])

    Returns:
        torch.Tensor: Result of the (un-)grouping operation.

    Raises:
        KeyError: If information about a dimension in `dim` is missing
            or can be removed. # noqa: DAR402
        RuntimeError: If the groups inferred from `equation` do not match
            the number of axes of `operand` # noqa: DAR402

    """
    dim = {} if dim is None else dim
    in_shape, out_shape, einsum_eq = _eingroup_preprocess(equation, operand, dim=dim)

    operand_in = operand.reshape(in_shape)
    result = torch.einsum(einsum_eq, operand_in)
    return result.reshape(out_shape)


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
    """Return shape the input needs to be reshaped, and the output shape."""

    def shape(groups, dim):
        return [group_dim(group, dim) for group in groups]

    def product(nums):
        assert len(nums) > 0

        result = 1
        for num in nums:
            result *= num
        return result

    def group_dim(group, dim):
        try:
            return product([dim[g] for g in group])
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
