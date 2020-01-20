import functools
from backpack.utils.einsum import einsum


def add_V_dim(old_mat):
    return old_mat.unsqueeze(-1)


def add_V_dim_new_convention(mat):
    return mat.unsqueeze(0)


def remove_V_dim_new_convention(mat):
    return mat.squeeze(0)


def remove_V_dim(old_mat):
    return old_mat.squeeze(-1)


def check_like_and_is_vec(mat_shape, like_shape):
    is_vec, fail = None, None
    if len(mat_shape) == len(like_shape):
        is_vec = True
        if not (mat_shape == like_shape):
            fail = True
    elif len(mat_shape) - len(like_shape) == 1:
        is_vec = False
        if not (mat_shape[1:] == like_shape):
            fail = True
    else:
        fail = True

    if fail:
        raise ValueError(
            "Accept {} or {}, got {}".format(like_shape, [-1, *like_shape], mat_shape)
        )

    return is_vec


def check_like_input_and_is_vec(module, mat):
    mat_shape = [int(dim) for dim in mat.shape]
    in_shape = [int(dim) for dim in module.input0_shape]

    return check_like_and_is_vec(mat_shape, in_shape)


def check_like_param_and_is_vec(module, mat, sum_batch, name):
    mat_shape = [int(dim) for dim in mat.shape]
    param_shape = [int(dim) for dim in getattr(module, name).shape]

    N = int(module.output_shape[0])
    out_shape = param_shape if sum_batch else [N, *param_shape]

    return check_like_and_is_vec(mat_shape, out_shape)


def hessian_matrix_product_accept_vectors(hessian_matrix_product):
    @functools.wraps(hessian_matrix_product)
    def wrapped_hessian_matrix_product(self, module, g_inp, g_out, **kwargs):

        hmp = hessian_matrix_product(self, module, g_inp, g_out)

        def new_hmp(mat):
            is_vec = check_like_input_and_is_vec(module, mat)
            mat_used = mat if not is_vec else add_V_dim_new_convention(mat)

            result = hmp(mat_used)

            check_like_input_and_is_vec(module, result)

            result = result if not is_vec else remove_V_dim_new_convention(result)

            return result

        return new_hmp

    return wrapped_hessian_matrix_product


def hessian_old_shape_convention(h_func):
    """Use old convention internally, new convention for IO."""

    @functools.wraps(h_func)
    def wrapped_h_use_old_convention(*args, **kwargs):
        print("[hessian]")

        result = h_func(*args, **kwargs)

        return einsum("nic->cni", result)

    return wrapped_h_use_old_convention
