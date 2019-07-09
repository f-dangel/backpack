from ...derivatives.mseloss import MSELossDerivatives
from ...derivatives.crossentropyloss import CrossEntropyLossDerivatives
from .cmpbase import CMPBase


class CMPLoss(CMPBase):
    def backpropagate(self, module, grad_input, grad_output):
        HMP = self.hessian_matrix_product(module, grad_input, grad_output)
        self.set_in_ctx(HMP)


class CMPMSELoss(CMPLoss, MSELossDerivatives):
    pass


class CMPCrossEntropyLoss(CMPLoss, CrossEntropyLossDerivatives):
    pass


EXTENSIONS = [CMPCrossEntropyLoss(), CMPMSELoss()]
