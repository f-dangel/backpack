from ...core.derivatives.mseloss import MSELossDerivatives
from ...core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from .kfacbase import KFACBase


class KFACLoss(KFACBase):
    def backpropagate(self, module, grad_input, grad_output):
        sqrt_H_sampled = self.sqrt_hessian_sampled(module, grad_input,
                                                   grad_output)
        self.set_in_ctx(sqrt_H_sampled)


class KFACMSELoss(KFACLoss, MSELossDerivatives):
    pass


class KFACCrossEntropyLoss(KFACLoss, CrossEntropyLossDerivatives):
    pass


EXTENSIONS = [KFACCrossEntropyLoss(), KFACMSELoss()]
