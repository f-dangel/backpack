from backpack.utils.utils import einsum


def matrix_from_kron_facs(factors):
    assert all_tensors_of_order(order=2, tensors=factors)
    mat = None
    for factor in factors:
        if mat is None:
            mat = factor
        else:
            new_shape = (mat.shape[0] * factor.shape[0],
                         mat.shape[1] * factor.shape[1])
            mat = einsum('ij,kl->ikjl',
                         (mat, factor)).contiguous().view(new_shape)
    return mat


def vp_from_kron_facs(factors):
    assert all_tensors_of_order(order=2, tensors=factors)

    shapes = [list(f.size()) for f in factors]
    _, col_dims = zip(*shapes)

    num_factors = len(shapes)
    equation = vp_einsum_equation(num_factors)

    def vp(v):
        assert len(v.shape) == 1
        v_reshaped = v.view(col_dims)
        return einsum(equation, *factors, v_reshaped).view(-1)

    return vp


def vp_einsum_equation(num_factors):
    letters = get_letters()
    in_str, v_str, out_str = "", "", ""

    for _ in range(num_factors):
        row_idx, col_idx = next(letters), next(letters)

        in_str += row_idx + col_idx + ","
        v_str += col_idx
        out_str += row_idx

    return "{}{}->{}".format(in_str, v_str, out_str)


def all_tensors_of_order(order, tensors):
    return all([len(t.shape) == order for t in tensors])


def get_letters(max_letters=26):
    for i in range(max_letters):
        yield chr(ord('a') + i)
