import functools


def kfacmp_unsqueeze_if_missing_dim(mat_dim):
    """
    Allows Kronecker-factored matrix-matrix routines to do matrix-vector products.
    """

    def kfacmp_wrapper(kfacmp):
        @functools.wraps(kfacmp)
        def wrapped_kfacmp_support_kfacvp(mat):
            is_vec = len(mat.shape) == mat_dim - 1
            mat_used = mat.unsqueeze(-1) if is_vec else mat
            result = kfacmp(mat_used)
            if is_vec:
                return result.squeeze(-1)
            else:
                return result

        return wrapped_kfacmp_support_kfacvp

    return kfacmp_wrapper
