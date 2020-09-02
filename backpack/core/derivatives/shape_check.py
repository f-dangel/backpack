"""
Helpers to support application of Jacobians to vectors
Helpers to check input and output sizes of Jacobian-matrix products.
"""
import functools


###############################################################################
#                              Utility functions                              #
###############################################################################
def add_V_dim(mat):
    return mat.unsqueeze(0)


def remove_V_dim(mat):
    if mat.shape[0] != 1:
        raise RuntimeError(
            "Cannot unsqueeze dimension 0. ", "Got tensor of shape {}".format(mat.shape)
        )
    return mat.squeeze(0)


def check_shape(mat, like, diff=1):
    """Compare dimension diff,diff+1, ... with dimension 0,1,..."""
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
    V1, V2 = mat1.shape[0], mat2.shape[0]
    if V1 != V2:
        raise RuntimeError("Number of vectors changed. Got {} and {}".format(V1, V2))


def check_like(mat, module, name, diff=1, *args, **kwargs):
    return check_shape(mat, getattr(module, name), diff=diff)


def check_like_with_sum_batch(mat, module, name, sum_batch=True, *args, **kwargs):
    diff = 1 if sum_batch else 2
    return check_shape(mat, getattr(module, name), diff=diff)


def same_dim_as(mat, module, name, *args, **kwargs):
    return len(mat.shape) == len(getattr(module, name).shape)


###############################################################################
#            Decorators for handling vectors as matrix special case           #
###############################################################################
def mat_prod_accept_vectors(mat_prod, vec_criterion):
    """Add support for vectors to matrix products.

    vec_criterion(mat, module) returns if mat is a vector.
    """

    @functools.wraps(mat_prod)
    def wrapped_mat_prod_accept_vectors(
        self, module, g_inp, g_out, mat, *args, **kwargs
    ):
        is_vec = vec_criterion(mat, module, *args, **kwargs)
        mat_in = mat if not is_vec else add_V_dim(mat)
        mat_out = mat_prod(self, module, g_inp, g_out, mat_in, *args, **kwargs)
        mat_out = mat_out if not is_vec else remove_V_dim(mat_out)

        return mat_out

    return wrapped_mat_prod_accept_vectors


# vec criteria
same_dim_as_output = functools.partial(same_dim_as, name="output")
same_dim_as_input = functools.partial(same_dim_as, name="input0")
same_dim_as_weight = functools.partial(same_dim_as, name="weight")
same_dim_as_bias = functools.partial(same_dim_as, name="bias")

# decorators for handling vectors
jac_t_mat_prod_accept_vectors = functools.partial(
    mat_prod_accept_vectors,
    vec_criterion=same_dim_as_output,
)

weight_jac_t_mat_prod_accept_vectors = functools.partial(
    mat_prod_accept_vectors,
    vec_criterion=same_dim_as_output,
)
bias_jac_t_mat_prod_accept_vectors = functools.partial(
    mat_prod_accept_vectors,
    vec_criterion=same_dim_as_output,
)

jac_mat_prod_accept_vectors = functools.partial(
    mat_prod_accept_vectors,
    vec_criterion=same_dim_as_input,
)

weight_jac_mat_prod_accept_vectors = functools.partial(
    mat_prod_accept_vectors,
    vec_criterion=same_dim_as_weight,
)

bias_jac_mat_prod_accept_vectors = functools.partial(
    mat_prod_accept_vectors,
    vec_criterion=same_dim_as_bias,
)


###############################################################################
#       Decorators for checking inputs and outputs of mat_prod routines       #
###############################################################################
def mat_prod_check_shapes(mat_prod, in_check, out_check):
    """Check that input and output have correct shapes."""

    @functools.wraps(mat_prod)
    def wrapped_mat_prod_check_shapes(self, module, g_inp, g_out, mat, *args, **kwargs):
        in_check(mat, module, *args, **kwargs)
        mat_out = mat_prod(self, module, g_inp, g_out, mat, *args, **kwargs)
        out_check(mat_out, module, *args, **kwargs)
        check_same_V_dim(mat_out, mat)

        return mat_out

    return wrapped_mat_prod_check_shapes


# input/output checker
shape_like_output = functools.partial(check_like, name="output")
shape_like_input = functools.partial(check_like, name="input0")
shape_like_weight = functools.partial(check_like, name="weight")
shape_like_bias = functools.partial(check_like, name="bias")
shape_like_weight_with_sum_batch = functools.partial(
    check_like_with_sum_batch, name="weight"
)
shape_like_bias_with_sum_batch = functools.partial(
    check_like_with_sum_batch, name="bias"
)

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


weight_jac_t_mat_prod_check_shapes = functools.partial(
    mat_prod_check_shapes,
    in_check=shape_like_output,
    out_check=shape_like_weight_with_sum_batch,
)
bias_jac_t_mat_prod_check_shapes = functools.partial(
    mat_prod_check_shapes,
    in_check=shape_like_output,
    out_check=shape_like_bias_with_sum_batch,
)

###############################################################################
#                     Wrapper for second-order extensions                     #
###############################################################################
residual_mat_prod_check_shapes = functools.partial(
    mat_prod_check_shapes, in_check=shape_like_output, out_check=shape_like_output
)

residual_mat_prod_accept_vectors = functools.partial(
    mat_prod_accept_vectors,
    vec_criterion=same_dim_as_input,
)


# TODO Refactor using partials
def make_hessian_mat_prod_accept_vectors(make_hessian_mat_prod):
    @functools.wraps(make_hessian_mat_prod)
    def wrapped_make_hessian_mat_prod(self, module, g_inp, g_out):

        hessian_mat_prod = make_hessian_mat_prod(self, module, g_inp, g_out)

        def new_hessian_mat_prod(mat):
            is_vec = same_dim_as(mat, module, "input0")
            mat_in = mat if not is_vec else add_V_dim(mat)
            mat_out = hessian_mat_prod(mat_in)
            mat_out = mat_out if not is_vec else remove_V_dim(mat_out)

            return mat_out

        return new_hessian_mat_prod

    return wrapped_make_hessian_mat_prod


def make_hessian_mat_prod_check_shapes(make_hessian_mat_prod):
    @functools.wraps(make_hessian_mat_prod)
    def wrapped_make_hessian_mat_prod(self, module, g_inp, g_out):

        hessian_mat_prod = make_hessian_mat_prod(self, module, g_inp, g_out)

        def new_hessian_mat_prod(mat):
            check_like(mat, module, "input0")
            result = hessian_mat_prod(mat)
            check_like(result, module, "input0")

            return result

        return new_hessian_mat_prod

    return wrapped_make_hessian_mat_prod
