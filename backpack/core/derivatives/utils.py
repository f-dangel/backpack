import functools


def unsqueeze_if_missing_dim(mat_dim):
    """Allow Jacobian-matrixi routines to do Jacobian-vector products."""

    def jmp_wrapper(jmp):
        @functools.wraps(jmp)
        def wrapped_jmp_support_jvp(self, module, grad_input, grad_output, mat,
                                    **kwargs):
            is_vec = (len(mat.shape) == mat_dim - 1)
            print("It's a vector")
            print(mat.shape)
            mat_used = mat.unsqueeze(-1) if is_vec else mat
            result = jmp(self, module, grad_input, grad_output, mat_used,
                         **kwargs)
            if is_vec:
                return result.squeeze(-1)
            else:
                return result

        return wrapped_jmp_support_jvp

    return jmp_wrapper
