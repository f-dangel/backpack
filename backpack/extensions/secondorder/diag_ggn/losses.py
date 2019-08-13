from backpack.core.derivatives.mseloss import MSELossDerivatives
from backpack.core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from backpack.extensions.secondorder.hbp import LossHessianStrategy

from .diag_ggn_base import DiagGGNBaseModule


class DiagGGNLoss(DiagGGNBaseModule):
    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):

        if ext.loss_hessian_strategy == LossHessianStrategy.EXACT:
            hess_func = self.derivatives.sqrt_hessian
        elif ext.loss_hessian_strategy == LossHessianStrategy.SAMPLING:
            hess_func = self.derivatives.sqrt_hessian_sampled
        else:
            raise ValueError(
                "Unknown hessian strategy {}".format(ext.loss_hessian_strategy)
            )

        return hess_func(module, grad_inp, grad_out)


class DiagGGNMSELoss(DiagGGNLoss):
    def __init__(self):
        super().__init__(derivatives=MSELossDerivatives())


class DiagGGNCrossEntropyLoss(DiagGGNLoss):
    def __init__(self):
        super().__init__(derivatives=CrossEntropyLossDerivatives())
