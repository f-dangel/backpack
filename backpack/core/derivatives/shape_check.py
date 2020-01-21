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


def check_like(mat, module, name, diff=1):
    return check_shape(mat, getattr(module, name), diff=diff)


def same_dim_as(mat, module, name):
    return len(mat.shape) == len(getattr(module, name).shape)


def one_more_dim_as(mat, module, name):
    return len(mat.shape) - len(getattr(module, name).shape) == 1


def jac_t_mat_prod_accept_vectors(jac_t_mat_prod):
    """Add support for vectors to Jᵀ-matrix products."""

    @functools.wraps(jac_t_mat_prod)
    def wrapped_jac_t_mat_prod(self, module, g_inp, g_out, mat):
        is_vec = same_dim_as(mat, module, "output")

        mat_in = mat if not is_vec else add_V_dim(mat)
        mat_out = jac_t_mat_prod(self, module, g_inp, g_out, mat_in)
        mat_out = mat_out if not is_vec else remove_V_dim(mat_out)

        return mat_out

    return wrapped_jac_t_mat_prod


def _param_jac_t_mat_prod_accept_vectors(jac_t_mat_prod):
    """Make transpose Jacobian w.r.t. parameter accept vectors."""

    def jac_t_mat_prod_accept_vectors(jac_t_mat_prod):
        """Add support for vectors to Jᵀ-matrix products."""

        @functools.wraps(jac_t_mat_prod)
        def wrapped_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
            is_vec = same_dim_as(mat, module, "output")

            mat_in = mat if not is_vec else add_V_dim(mat)
            mat_out = jac_t_mat_prod(
                self, module, g_inp, g_out, mat_in, sum_batch=sum_batch
            )
            mat_out = mat_out if not is_vec else remove_V_dim(mat_out)

            return mat_out

        return wrapped_jac_t_mat_prod

    return jac_t_mat_prod_accept_vectors(jac_t_mat_prod)


def weight_jac_t_mat_prod_accept_vectors(weight_jac_t_mat_prod):
    """Make transpose Jacobian w.r.t. weight accept vectors."""
    return _param_jac_t_mat_prod_accept_vectors(weight_jac_t_mat_prod)


def bias_jac_t_mat_prod_accept_vectors(bias_jac_t_mat_prod):
    """Make transpose Jacobian w.r.t. bias accept vectors."""
    return _param_jac_t_mat_prod_accept_vectors(bias_jac_t_mat_prod)


def _param_jac_mat_prod_accept_vectors(jac_mat_prod, param):
    """Make Jacobian w.r.t. parameter accept vectors."""

    def jac_mat_prod_accept_vectors(jac_mat_prod):
        """Add support for vectors to J-matrix products."""

        @functools.wraps(jac_mat_prod)
        def wrapped_jac_mat_prod(self, module, g_inp, g_out, mat):
            is_vec = same_dim_as(mat, module, param)

            mat_in = mat if not is_vec else add_V_dim(mat)
            mat_out = jac_mat_prod(self, module, g_inp, g_out, mat_in)
            mat_out = mat_out if not is_vec else remove_V_dim(mat_out)

            return mat_out

        return wrapped_jac_mat_prod

    return jac_mat_prod_accept_vectors(jac_mat_prod)


def jac_mat_prod_accept_vectors(jac_mat_prod):
    """Make Jacobian w.r.t. input accept vectors."""
    return _param_jac_mat_prod_accept_vectors(jac_mat_prod, "input0")


def weight_jac_mat_prod_accept_vectors(weight_jac_mat_prod):
    """Make Jacobian w.r.t. weight accept vectors."""
    return _param_jac_mat_prod_accept_vectors(weight_jac_mat_prod, "weight")


def bias_jac_mat_prod_accept_vectors(bias_jac_mat_prod):
    """Make Jacobian w.r.t. bias accept vectors."""
    return _param_jac_mat_prod_accept_vectors(bias_jac_mat_prod, "bias")


def _param_jac_mat_prod_check_shapes(jac_mat_prod, param):
    """Check that input and output have correct shapes."""

    def jac_mat_prod_check_shapes(jac_mat_prod):
        @functools.wraps(jac_mat_prod)
        def wrapped_jac_mat_prod(self, module, g_inp, g_out, mat):
            check_like(mat, module, param)
            mat_out = jac_mat_prod(self, module, g_inp, g_out, mat)
            check_like(mat_out, module, "output")
            check_same_V_dim(mat_out, mat)

            return mat_out

        return wrapped_jac_mat_prod

    return jac_mat_prod_check_shapes(jac_mat_prod)


def bias_jac_mat_prod_check_shapes(bias_jac_mat_prod):
    return _param_jac_mat_prod_check_shapes(bias_jac_mat_prod, "bias")


def weight_jac_mat_prod_check_shapes(weight_jac_mat_prod):
    return _param_jac_mat_prod_check_shapes(weight_jac_mat_prod, "weight")


def jac_mat_prod_check_shapes(jac_mat_prod):
    return _param_jac_mat_prod_check_shapes(jac_mat_prod, "input0")


def jac_t_mat_prod_check_shapes(jac_t_mat_prod):
    """Check that input and output have correct shapes."""

    @functools.wraps(jac_t_mat_prod)
    def wrapped_jac_t_mat_prod(self, module, g_inp, g_out, mat):
        check_like(mat, module, "output")
        mat_out = jac_t_mat_prod(self, module, g_inp, g_out, mat)
        check_like(mat_out, module, "input0")
        check_same_V_dim(mat, mat_out)

        return mat_out

    return wrapped_jac_t_mat_prod


def _param_jac_t_mat_prod_check_shapes(jac_t_mat_prod, param):
    """Check that input and output have correct shapes."""

    def jac_t_mat_prod_check_shapes(jac_t_mat_prod):
        @functools.wraps(jac_t_mat_prod)
        def wrapped_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
            check_like(mat, module, "output")
            mat_out = jac_t_mat_prod(
                self, module, g_inp, g_out, mat, sum_batch=sum_batch
            )
            diff = 1 if sum_batch else 2
            print(diff, sum_batch, mat_out.shape, getattr(module, param).shape)
            check_like(mat_out, module, param, diff=diff)
            check_same_V_dim(mat_out, mat)

            return mat_out

        return wrapped_jac_t_mat_prod

    return jac_t_mat_prod_check_shapes(jac_t_mat_prod)


def bias_jac_t_mat_prod_check_shapes(bias_jac_t_mat_prod):
    return _param_jac_t_mat_prod_check_shapes(bias_jac_t_mat_prod, "bias")


def weight_jac_t_mat_prod_check_shapes(weight_jac_t_mat_prod):
    return _param_jac_t_mat_prod_check_shapes(weight_jac_t_mat_prod, "weight")


def R_mat_prod_check_shapes(make_R_mat_prod):
    """Check that input and output have correct shapes."""

    @functools.wraps(make_R_mat_prod)
    def wrapped_make_R_mat_prod(self, module, g_inp, g_out):
        def checked_R_mat_prod(mat):
            check_like(mat, module, "input0")
            mat_out = make_R_mat_prod(self, module, g_inp, g_out)(mat)
            check_like(mat_out, module, "input0")
            check_same_V_dim(mat, mat_out)

            return mat_out

        return checked_R_mat_prod

    return wrapped_make_R_mat_prod


def R_mat_prod_accept_vectors(make_R_mat_prod):
    """Add support for vectors to Residual-matrix products."""

    @functools.wraps(make_R_mat_prod)
    def wrapped_make_R_mat_prod(self, module, g_inp, g_out):
        def new_R_mat_prod(mat):
            is_vec = same_dim_as(mat, module, "input0")
            mat_in = mat if not is_vec else add_V_dim(mat)
            mat_out = make_R_mat_prod(self, module, g_inp, g_out)(mat_in)
            mat_out = mat_out if not is_vec else remove_V_dim(mat_out)

            return mat_out

        return new_R_mat_prod

    return wrapped_make_R_mat_prod


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
