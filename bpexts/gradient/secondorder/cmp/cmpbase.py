from ...backpropextension import BackpropExtension
from ...ctxinteract import ActOnCTX
from ...extensions import CMP

BACKPROPAGATED_MP_NAME = "_cmp_backpropagated_mp"
EXTENSION = CMP


class CMPBase(BackpropExtension, ActOnCTX):
    """Given matrix-vector product routine `MVP(A)` backpropagate
     to `MVP(J^T A J)`."""

    def __init__(self, params=None):
        if params is None:
            params = []
        ActOnCTX.__init__(self, BACKPROPAGATED_MP_NAME)
        BackpropExtension.__init__(
            self, self.get_module(), EXTENSION, params=params)

    def backpropagate(self, module, grad_input, grad_output):
        CMP_out = self.get_from_ctx()

        def CMP_in(mat):
            print(mat.size())
            Jmat = self.jac_mat_prod(module, grad_input, grad_output, mat)
            print(Jmat.size())
            CJmat = CMP_out(Jmat)
            print(CJmat.size())
            JTCJmat = self.jac_t_mat_prod(module, grad_input, grad_output,
                                          CJmat)
            print(JTCJmat.size())
            return JTCJmat

        self.set_in_ctx(CMP_in)
