from ...backpropextension import BackpropExtension
from ...ctxinteract import ActOnCTX
from ...extensions import CMP
from ....utils import einsum

BACKPROPAGATED_MP_NAME = "_cmp_backpropagated_mp"
EXTENSION = CMP

# Different curvature matrices
HESSIAN = 'hessian'
GGN = 'ggn'
PCH_ABS = 'pch-abs'
PCH_CLIP = 'pch-clip'


# Actions on residual terms
def nothing(res):
    return res


def to_zero(res):
    # return res.zero()
    return None


def remove_negative_values(res):
    return res.clamp(min=0)


def to_abs(res):
    return res.abs()


class CMPBase(BackpropExtension, ActOnCTX):
    """Given matrix-vector product routine `MVP(A)` backpropagate
     to `MVP(J^T A J)`."""

    # TODO: Move to own class
    CURVATURE_CHOICES = [
        HESSIAN,
        GGN,
        PCH_CLIP,
        PCH_ABS,
    ]
    REQUIRE_PSD_LOSS_HESSIAN = {
        HESSIAN: False,
        GGN: True,
        PCH_ABS: True,
        PCH_CLIP: True,
    }
    REQUIRE_RESIDUAL = {
        HESSIAN: True,
        GGN: False,
        PCH_ABS: True,
        PCH_CLIP: True,
    }
    RESIDUAL_ACTIONS = {
        HESSIAN: nothing,
        GGN: to_zero,
        PCH_ABS: to_abs,
        PCH_CLIP: remove_negative_values,
    }

    def __init__(self, params=None):
        if params is None:
            params = []
        ActOnCTX.__init__(self, BACKPROPAGATED_MP_NAME)
        BackpropExtension.__init__(
            self, self.get_module(), EXTENSION, params=params)

    def backpropagate(self, module, grad_input, grad_output):
        CMP_out = self.get_from_ctx()

        # second-order module effects
        residual = self._compute_residual_diag_if_nonzero(
            module, grad_input, grad_output)

        def CMP_in(mat, which):
            """Multiplication of curvature matrix with matrix `mat`.

            Parameters:
            -----------
            mat : torch.Tensor
                Matrix that will be multiplied.
            which : str
                Which curvature matrix to use. For choices,
                Choices: See `CURVATURE_CHOICES`
            """
            self._check_curvature_exists(which)

            Jmat = self.jac_mat_prod(module, grad_input, grad_output, mat)
            CJmat = CMP_out(Jmat, which=which)
            JTCJmat = self.jac_t_mat_prod(module, grad_input, grad_output,
                                          CJmat)

            # None if zero or curvature neglects 2nd-order module effects
            res_mod = self._modify_residual_for_curvature(residual, which)
            if res_mod is not None:
                JTCJmat.add_(einsum('bi,bic->bic', (res_mod, mat)))

            return JTCJmat

        self.set_in_ctx(CMP_in)

    def _compute_residual_diag_if_nonzero(self, module, grad_input,
                                          grad_output):
        if self.hessian_is_zero():
            return None

        if not self.hessian_is_diagonal():
            raise AttributeError(
                "Residual terms are only supported for elementwise functions")

        # second order module effects
        return self.hessian_diagonal(module, grad_input, grad_output)

    def _modify_residual_for_curvature(self, res, which):
        if res is None:
            return None
        else:
            return self.RESIDUAL_ACTIONS[which](res)

    def _check_curvature_exists(self, which):
        if not which in self.CURVATURE_CHOICES:
            raise AttributeError(
                "Unknown curvature matrix: {}.\n Expecting one of {}".format(
                    which, self.CURVATURE_CHOICES))
