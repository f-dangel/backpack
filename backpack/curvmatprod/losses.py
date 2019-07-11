from ..core.derivatives.mseloss import MSELossDerivatives
from ..core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from .cmpbase import CMPBase
from ..curvature import Curvature


class CMPLoss(CMPBase):
    def backpropagate(self, module, grad_input, grad_output):
        Curvature.check_loss_hessian(self.hessian_is_psd())

        CMP = self.hessian_matrix_product(module, grad_input, grad_output)
        self.set_in_ctx(CMP)


class CMPMSELoss(CMPLoss, MSELossDerivatives):
    pass


class CMPCrossEntropyLoss(CMPLoss, CrossEntropyLossDerivatives):
    pass


EXTENSIONS = [CMPCrossEntropyLoss(), CMPMSELoss()]
