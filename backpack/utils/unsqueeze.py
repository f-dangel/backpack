import functools


def jmp_unsqueeze_if_missing_dim(mat_dim):
    """Allow Jacobian-matrix routines to do Jacobian-vector products."""

    def jmp_wrapper(jmp):
        @functools.wraps(jmp)
        def wrapped_jmp_support_jvp(self, module, g_inp, g_out, mat, **kwargs):
            is_vec = len(mat.shape) == mat_dim - 1
            mat_used = mat.unsqueeze(-1) if is_vec else mat
            result = jmp(self, module, g_inp, g_out, mat_used, **kwargs)
            if is_vec:
                return result.squeeze(-1)
            else:
                return result

        return wrapped_jmp_support_jvp

    return jmp_wrapper


def hmp_unsqueeze_if_missing_dim(mat_dim):
    """Allow Hessian-matrix routines to do Hessian-vector products."""

    def hmp_wrapper(hmp):
        @functools.wraps(hmp)
        def wrapped_hmp_support_hvp(mat):
            is_vec = len(mat.shape) == mat_dim - 1
            mat_used = mat.unsqueeze(-1) if is_vec else mat
            result = hmp(mat_used)
            if is_vec:
                return result.squeeze(-1)
            else:
                return result

        return wrapped_hmp_support_hvp

    return hmp_wrapper


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
