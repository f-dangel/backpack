"""
Helpers to support application of Jacobians to vectors
Helpers to check input and output sizes of Jacobian-matrix products.
"""
import functools


def add_V_dim(mat):
    return mat.unsqueeze(0)


def remove_V_dim(mat):
    if mat.shape[0] != 1:
        raise RuntimeError(
            "Cannot unsqueeze dimension 0. ", "Got tensor of shape {}".format(mat.shape)
        )
    return mat.squeeze(0)


def check_shape(mat, like):
    """Compare dimension 1,2, ... with dimension 0,1,..."""
    mat_shape = [int(dim) for dim in mat.shape]
    like_shape = [int(dim) for dim in like.shape]

    if len(mat_shape) - len(like_shape) != 1:
        raise RuntimeError(
            "Difference in dimension must be 1.",
            " Got {} and {}".format(mat_shape, like_shape),
        )
    if mat_shape[1:] != like_shape:
        raise RuntimeError(
            "Compared shapes {} and {} do not match. ".format(
                mat_shape[1:], like_shape
            ),
            "Got {} and {}".format(mat_shape, like_shape),
        )


def check_same_V_dim(mat1, mat2):
    V1, V2 = mat1.shape[0], mat2.shape[0]
    if V1 != V2:
        raise RuntimeError("Number of vectors changed. Got {} and {}".format(V1, V2))


def check_like_output(module, mat):
    print("Check output [{}]".format(module))
    return check_shape(mat, module.output)


def check_like_input(module, mat):
    return check_shape(mat, module.input0)


def jac_t_mat_prod_accept_vectors(jac_t_mat_prod):
    """Add support for vectors to Jᵀ-matrix products."""

    def same_dim_as_output(mat, module):
        return len(mat.shape) == len(module.output.shape)

    @functools.wraps(jac_t_mat_prod)
    def wrapped_jac_t_mat_prod(self, module, g_inp, g_out, mat):
        is_vec = same_dim_as_output(mat, module)

        mat_in = mat if not is_vec else add_V_dim(mat)
        mat_out = jac_t_mat_prod(self, module, g_inp, g_out, mat_in)
        mat_out = mat_out if not is_vec else remove_V_dim(mat_out)

        return mat_out

    return wrapped_jac_t_mat_prod


def jac_t_mat_prod_check_shapes(jac_t_mat_prod):
    """Check that input and output have correct shapes."""

    @functools.wraps(jac_t_mat_prod)
    def wrapped_jac_t_mat_prod(self, module, g_inp, g_out, mat):
        check_like_output(module, mat)
        mat_out = jac_t_mat_prod(self, module, g_inp, g_out, mat)
        check_like_input(module, mat_out)
        check_same_V_dim(mat_out, mat)

        return mat_out

    return wrapped_jac_t_mat_prod


def jac_mat_prod_accept_vectors(jac_mat_prod):
    """Add support for vectors to J-matrix products."""

    def same_dim_as_input(mat, module):
        return len(mat.shape) == len(module.input0.shape)

    @functools.wraps(jac_mat_prod)
    def wrapped_jac_mat_prod(self, module, g_inp, g_out, mat):
        is_vec = same_dim_as_input(mat, module)

        mat_in = mat if not is_vec else add_V_dim(mat)
        mat_out = jac_mat_prod(self, module, g_inp, g_out, mat_in)
        mat_out = mat_out if not is_vec else remove_V_dim(mat_out)

        return mat_out

    return wrapped_jac_mat_prod


def jac_mat_prod_check_shapes(jac_mat_prod):
    """Check that input and output have correct shapes."""

    @functools.wraps(jac_mat_prod)
    def wrapped_jac_mat_prod(self, module, g_inp, g_out, mat):
        check_like_input(module, mat)
        mat_out = jac_mat_prod(self, module, g_inp, g_out, mat)
        check_like_output(module, mat_out)
        check_same_V_dim(mat_out, mat)

        return mat_out

    return wrapped_jac_mat_prod
