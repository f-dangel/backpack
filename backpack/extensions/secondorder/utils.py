from backpack.core.derivatives.utils import kfacmp_unsqueeze_if_missing_dim
from backpack.utils.utils import einsum


def kfacs_to_mat(factors):
    """Given [A, B, C, ...], return A ⊗ B ⊗ C ⊗ ... ."""
    mat = None
    for factor in factors:
        if mat is None:
            assert is_matrix(factor)
            mat = factor
        else:
            mat = two_kfacs_to_mat(mat, factor)

    return mat


def two_kfacs_to_mat(A, B):
    """Given A, B, return A ⊗ B."""
    assert is_matrix(A)
    assert is_matrix(B)

    mat_shape = (
        A.shape[0] * B.shape[0],
        A.shape[1] * B.shape[1],
    )
    mat = einsum("ij,kl->ikjl", (A, B)).contiguous().view(mat_shape)
    return mat


def kfac_mat_prod(factors):
    """Return function v ↦ (A ⊗ B ⊗ ...)v for `factors = [A, B, ...]` """
    assert all_tensors_of_order(order=2, tensors=factors)

    shapes = [list(f.size()) for f in factors]
    _, col_dims = zip(*shapes)

    num_factors = len(shapes)
    equation = kfac_mat_prod_einsum_equation(num_factors)

    @kfacmp_unsqueeze_if_missing_dim(mat_dim=2)
    def kfacmp(mat):
        assert is_matrix(mat)
        _, mat_cols = mat.shape
        mat_reshaped = mat.view(*(col_dims), mat_cols)
        return einsum(equation, mat_reshaped, *factors).contiguous().view(-1, mat_cols)

    return kfacmp


def apply_kfac_mat_prod(factors, mat):
    """Return (A ⊗ B ⊗ ...) mat for `factors = [A, B, ...]`

    All Kronecker factors have to be matrices.
    """
    kfacmp = kfac_mat_prod(factors)
    return kfacmp(mat)


def kfac_mat_prod_einsum_equation(num_factors):
    letters = get_letters()
    in_str, mat_str, out_str = "", "", ""

    for _ in range(num_factors):
        row_idx, col_idx = next(letters), next(letters)

        in_str += "," + row_idx + col_idx
        mat_str += col_idx
        out_str += row_idx

    mat_col_idx = next(letters)
    mat_str += mat_col_idx
    out_str += mat_col_idx

    return "{}{}->{}".format(mat_str, in_str, out_str)


def all_tensors_of_order(order, tensors):
    return all([is_tensor_of_order(order, t) for t in tensors])


def is_tensor_of_order(order, tensor):
    return len(tensor.shape) == order


def is_matrix(tensor):
    matrix_order = 2
    return is_tensor_of_order(matrix_order, tensor)


def is_vector(tensor):
    vector_order = 1
    return is_tensor_of_order(vector_order, tensor)


def get_letters(max_letters=26):
    for i in range(max_letters):
        yield chr(ord("a") + i)
