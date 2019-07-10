from ...derivatives.mseloss import MSELossDerivatives
from ...derivatives.crossentropyloss import CrossEntropyLossDerivatives
from .cmpbase import CMPBase
from ..curvature import Curvature


class CMPLoss(CMPBase):
    def backpropagate(self, module, grad_input, grad_output):
        hmp = self.hessian_matrix_product(module, grad_input, grad_output)

        def CMP(mat, which=Curvature.HESSIAN):
            Curvature.check_loss_hessian(which, self.hessian_is_psd())
            return hmp(mat)

        self.set_in_ctx(CMP)


class CMPMSELoss(CMPLoss, MSELossDerivatives):
    pass


class CMPCrossEntropyLoss(CMPLoss, CrossEntropyLossDerivatives):
    pass


EXTENSIONS = [CMPCrossEntropyLoss(), CMPMSELoss()]
