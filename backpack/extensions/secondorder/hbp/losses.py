from backpack.core.derivatives.mseloss import MSELossDerivatives
from backpack.core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from .hbp_options import LossHessianStrategy
from backpack.extensions.curvature import Curvature
from .hbpbase import HBPBaseModule


class HBPLoss(HBPBaseModule):
    def __init__(self, derivatives, params=None):
        super().__init__(derivatives=derivatives, params=params)

        self.LOSS_HESSIAN_GETTERS = {
            LossHessianStrategy.EXACT: self.derivatives.sqrt_hessian,
            LossHessianStrategy.SAMPLING: self.derivatives.sqrt_hessian_sampled,
            LossHessianStrategy.AVERAGE: self.derivatives.sum_hessian,
        }

    def backpropagate(self, ext, module, g_inp, g_out, backproped):
        Curvature.check_loss_hessian(
            self.derivatives.hessian_is_psd(),
            curv_type=ext.get_curv_type()
        )

        hessian_strategy = ext.get_loss_hessian_strategy()
        H_func = self.LOSS_HESSIAN_GETTERS[hessian_strategy]
        H_loss = H_func(module, g_inp, g_out)

        return H_loss


class HBPMSELoss(HBPLoss):
    def __init__(self):
        super().__init__(derivatives=MSELossDerivatives())


class HBPCrossEntropyLoss(HBPLoss):
    def __init__(self):
        super().__init__(derivatives=CrossEntropyLossDerivatives())
