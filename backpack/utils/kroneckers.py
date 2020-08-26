from torch import einsum

from backpack.utils.unsqueeze import kfacmp_unsqueeze_if_missing_dim


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


def inv_kfac_mat_prod(factors, shift=None):
    """Return function M ↦ [(A + 𝜆₁I)⁻¹ ⊗ (A + 𝜆₂I)⁻¹⊗ ...] M
    given [A, B, ...], [𝜆₁, 𝜆₂, ...].
    """
    inv_factors = inv_kfacs(factors, shift)
    return kfac_mat_prod(inv_factors)


def apply_inv_kfac_mat_prod(factors, mat, shift=None):
    """Return [(A + 𝜆₁I)⁻¹ ⊗ (A + 𝜆₂I)⁻¹⊗ ...] M."""
    inv_mat_prod = inv_kfac_mat_prod(factors, shift)
    return inv_mat_prod(mat)


def inv_kfacs(factors, shift=None):
    """Given [A, B, ...], [𝜆₁, 𝜆₂, ...] Return [(A + 𝜆₁I)⁻¹, (A + 𝜆₂I)⁻¹, ...].

    I denotes the identity matrix. All KFACs are assumed symmetric.

    Parameters:
    -----------
    shift: list, tuple, float:
        Diagonal shift of the eigenvalues. Per default, no shift is applied.
        If float, the same shift is applied to all factors.
    """

    def make_shifts():
        """Turn user-specified shift into a value for each factor."""
        same = shift is None or isinstance(shift, float)
        if same:
            value = 0.0 if shift is None else shift
            return [value for factor in factors]
        else:
            assert isinstance(shift, (tuple, list))
            assert len(factors) == len(shift)
            return shift

    def sym_mat_inv(mat, shift, truncate=1e-8):
        """Inverse of a symmetric matrix A -> (A + 𝜆I)⁻¹.

        Computed by eigenvalue decomposition. Eigenvalues with small
        absolute values are truncated.
        """
        eigvals, eigvecs = mat.symeig(eigenvectors=True)
        eigvals.add_(shift)
        inv_eigvals = 1.0 / eigvals
        inv_truncate = 1.0 / truncate
        inv_eigvals.clamp_(min=-inv_truncate, max=inv_truncate)
        return einsum("ij,j,kj->ik", (eigvecs, inv_eigvals, eigvecs))

    shifts = make_shifts()
    return [sym_mat_inv(mat, shift) for mat, shift in zip(factors, shifts)]


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
    return all(is_tensor_of_order(order, t) for t in tensors)


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
