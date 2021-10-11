"""Helpers to support application of Jacobians to vectors.

Helpers to check input and output sizes of Jacobian-matrix products.
"""
import functools
from typing import Any, Callable

from torch import Tensor
from torch.nn import Module

from backpack.utils.subsampling import subsample


###############################################################################
#                              Utility functions                              #
###############################################################################
def _add_V_dim(mat):
    return mat.unsqueeze(0)


def _remove_V_dim(mat):
    if mat.shape[0] != 1:
        raise RuntimeError(
            "Cannot unsqueeze dimension 0. ", "Got tensor of shape {}".format(mat.shape)
        )
    return mat.squeeze(0)


def check_shape(mat: Tensor, like: Tensor, diff: int = 1) -> None:
    """Compare dimension diff,diff+1, ... with dimension 0,1,...

    Args:
        mat: matrix
        like: comparison matrix
        diff: difference in dimensions. Defaults to 1.

    Raises:
        RuntimeError: if shape does not fit
    """
    mat_shape = [int(dim) for dim in mat.shape]
    like_shape = [int(dim) for dim in like.shape]

    if len(mat_shape) - len(like_shape) != diff:
        raise RuntimeError(
            "Difference in dimension must be {}.".format(diff),
            " Got {} and {}".format(mat_shape, like_shape),
        )
    if mat_shape[diff:] != like_shape:
        raise RuntimeError(
            "Compared shapes {} and {} do not match. ".format(
                mat_shape[diff:], like_shape
            ),
            "Got {} and {}".format(mat_shape, like_shape),
        )


def check_same_V_dim(mat1, mat2):
    """Check whether V dim (first dim) matches.

    Args:
        mat1: first tensor
        mat2: second tensor

    Raises:
        RuntimeError: if V dim (first dim) doesn't match
    """
    V1, V2 = mat1.shape[0], mat2.shape[0]
    if V1 != V2:
        raise RuntimeError("Number of vectors changed. Got {} and {}".format(V1, V2))


def _check_like(mat, module, name, diff=1, *args, **kwargs):
    if name in ["output", "input0"] and "subsampling" in kwargs.keys():
        compare = subsample(
            getattr(module, name), dim=0, subsampling=kwargs["subsampling"]
        )
    else:
        compare = getattr(module, name)

    return check_shape(mat, compare, diff=diff)


def check_like_with_sum_batch(mat, module, name, sum_batch=True, *args, **kwargs):
    """Checks shape, considers sum_batch.

    Args:
        mat: matrix to multiply
        module: module
        name: parameter to operate on: module.name
        sum_batch: whether to consider with or without sum
        *args: ignored
        **kwargs: ignored
    """
    diff = 1 if sum_batch else 2
    check_shape(mat, getattr(module, name), diff=diff)


def _same_dim_as(mat, module, name, *args, **kwargs):
    return len(mat.shape) == len(getattr(module, name).shape)


###############################################################################
#            Decorators for handling vectors as matrix special case           #
###############################################################################
def _mat_prod_accept_vectors(
    mat_prod: Callable[..., Tensor],
    vec_criterion: Callable[[Tensor, Module, Any, Any], bool],
) -> Callable[..., Tensor]:
    """Add support for vectors to matrix products.

    vec_criterion(mat, module) returns if mat is a vector.

    Args:
        mat_prod: Function that processes multiple vectors in format of a matrix.
        vec_criterion: Function that returns true if an input is a single vector
            that must be formatted into a matrix first before processing.

    Returns:
        Wrapped ``mat_prod`` function that processes multiple vectors in format of
            a matrix, and supports vector-shaped inputs which are internally converted
            to the correct format.
            Preserves format of input:
                If the input format is a vector, the output format is a vector.
                If the input format is a matrix, the output format is a matrix.
    """

    @functools.wraps(mat_prod)
    def _wrapped_mat_prod_accept_vectors(
        self, module, g_inp, g_out, mat, *args, **kwargs
    ):
        is_vec = vec_criterion(mat, module, *args, **kwargs)
        mat_in = mat if not is_vec else _add_V_dim(mat)
        mat_out = mat_prod(self, module, g_inp, g_out, mat_in, *args, **kwargs)
        mat_out = mat_out if not is_vec else _remove_V_dim(mat_out)

        return mat_out

    return _wrapped_mat_prod_accept_vectors


# vec criteria
same_dim_as_output = functools.partial(_same_dim_as, name="output")
same_dim_as_input = functools.partial(_same_dim_as, name="input0")
same_dim_as_weight = functools.partial(_same_dim_as, name="weight")
same_dim_as_bias = functools.partial(_same_dim_as, name="bias")

# decorators for handling vectors
jac_t_mat_prod_accept_vectors = functools.partial(
    _mat_prod_accept_vectors,
    vec_criterion=same_dim_as_output,
)

jac_mat_prod_accept_vectors = functools.partial(
    _mat_prod_accept_vectors,
    vec_criterion=same_dim_as_input,
)

weight_jac_mat_prod_accept_vectors = functools.partial(
    _mat_prod_accept_vectors,
    vec_criterion=same_dim_as_weight,
)

bias_jac_mat_prod_accept_vectors = functools.partial(
    _mat_prod_accept_vectors,
    vec_criterion=same_dim_as_bias,
)


###############################################################################
#       Decorators for checking inputs and outputs of mat_prod routines       #
###############################################################################
def mat_prod_check_shapes(
    mat_prod: Callable, in_check: Callable, out_check: Callable
) -> Callable[..., Tensor]:
    """Check that input and output have correct shapes.

    Args:
        mat_prod: Function that applies a derivative operator to multiple vectors
            handed in as a matrix.
        in_check: Function that checks the input to mat_prod
        out_check: Function that checks the output to mat_prod

    Returns:
        Wrapped mat_prod function with input and output checks
    """

    @functools.wraps(mat_prod)
    def wrapped_mat_prod_check_shapes(self, module, g_inp, g_out, mat, *args, **kwargs):
        in_check(mat, module, *args, **kwargs)
        mat_out = mat_prod(self, module, g_inp, g_out, mat, *args, **kwargs)
        out_check(mat_out, module, *args, **kwargs)
        check_same_V_dim(mat_out, mat)

        return mat_out

    return wrapped_mat_prod_check_shapes


# input/output checker
shape_like_output = functools.partial(_check_like, name="output")
shape_like_input = functools.partial(_check_like, name="input0")
shape_like_weight = functools.partial(_check_like, name="weight")
shape_like_bias = functools.partial(_check_like, name="bias")

# decorators for shape checking
jac_mat_prod_check_shapes = functools.partial(
    mat_prod_check_shapes, in_check=shape_like_input, out_check=shape_like_output
)

weight_jac_mat_prod_check_shapes = functools.partial(
    mat_prod_check_shapes, in_check=shape_like_weight, out_check=shape_like_output
)

bias_jac_mat_prod_check_shapes = functools.partial(
    mat_prod_check_shapes, in_check=shape_like_bias, out_check=shape_like_output
)

jac_t_mat_prod_check_shapes = functools.partial(
    mat_prod_check_shapes, in_check=shape_like_output, out_check=shape_like_input
)

###############################################################################
#                     Wrapper for second-order extensions                     #
###############################################################################
residual_mat_prod_check_shapes = functools.partial(
    mat_prod_check_shapes, in_check=shape_like_output, out_check=shape_like_output
)

residual_mat_prod_accept_vectors = functools.partial(
    _mat_prod_accept_vectors,
    vec_criterion=same_dim_as_input,
)


# TODO Refactor using partials
def make_hessian_mat_prod_accept_vectors(
    make_hessian_mat_prod: Callable,
) -> Callable[..., Callable[..., Tensor]]:
    """Accept vectors for hessian_mat_prod.

    Args:
        make_hessian_mat_prod: Function that creates multiplication routine
            of a matrix with the module Hessian

    Returns:
        Wrapped hessian_mat_prod which converts vector-format inputs to a matrix
            before processing. Preserves format of input.
    """

    @functools.wraps(make_hessian_mat_prod)
    def _wrapped_make_hessian_mat_prod(self, module, g_inp, g_out):

        hessian_mat_prod = make_hessian_mat_prod(self, module, g_inp, g_out)

        def _new_hessian_mat_prod(mat):
            is_vec = _same_dim_as(mat, module, "input0")
            mat_in = mat if not is_vec else _add_V_dim(mat)
            mat_out = hessian_mat_prod(mat_in)
            mat_out = mat_out if not is_vec else _remove_V_dim(mat_out)

            return mat_out

        return _new_hessian_mat_prod

    return _wrapped_make_hessian_mat_prod


def make_hessian_mat_prod_check_shapes(
    make_hessian_mat_prod: Callable[..., Callable[..., Tensor]],
) -> Callable[..., Callable[..., Tensor]]:
    """Wrap hessian_mat_prod with shape checks for input and output.

    Args:
        make_hessian_mat_prod: function that creates multiplication routine of
            a matrix with the module Hessian.

    Returns:
        wrapped hessian_mat_prod with shape checks for input and output
    """

    @functools.wraps(make_hessian_mat_prod)
    def _wrapped_make_hessian_mat_prod(self, module, g_inp, g_out):

        hessian_mat_prod = make_hessian_mat_prod(self, module, g_inp, g_out)

        def _new_hessian_mat_prod(mat):
            _check_like(mat, module, "input0")
            result = hessian_mat_prod(mat)
            _check_like(result, module, "input0")

            return result

        return _new_hessian_mat_prod

    return _wrapped_make_hessian_mat_prod


def param_mjp_accept_vectors(mat_prod: Callable[..., Tensor]) -> Callable[..., Tensor]:
    """Add support for vectors to matrix products.

    vec_criterion(mat, module) returns if mat is a vector.

    Args:
        mat_prod: Function that processes multiple vectors in format of a matrix.

    Returns:
        Wrapped ``mat_prod`` function that processes multiple vectors in format of
        a matrix, and supports vector-shaped inputs which are internally converted
        to the correct format.
        Preserves format of input:
            If the input format is a vector, the output format is a vector.
            If the input format is a matrix, the output format is a matrix.
    """

    @functools.wraps(mat_prod)
    def _wrapped_mat_prod_accept_vectors(
        self, param_str, module, g_inp, g_out, mat, *args, **kwargs
    ):
        is_vec = same_dim_as_output(mat, module)
        mat_in = mat if not is_vec else _add_V_dim(mat)
        mat_out = mat_prod(
            self, param_str, module, g_inp, g_out, mat_in, *args, **kwargs
        )
        mat_out = mat_out if not is_vec else _remove_V_dim(mat_out)

        return mat_out

    return _wrapped_mat_prod_accept_vectors
