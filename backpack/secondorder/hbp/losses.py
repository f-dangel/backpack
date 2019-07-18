from ...core.derivatives.mseloss import MSELossDerivatives
from ...core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from ..strategies import LossHessianStrategy
from .hbpbase import HBPBase
from ...curvature import Curvature


class HBPLoss(HBPBase):
    def __init__(self, params=None):
        super().__init__(params=params)

        self.LOSS_HESSIAN_GETTERS = {
            LossHessianStrategy.EXACT: self.sqrt_hessian,
            LossHessianStrategy.SAMPLING: self.sqrt_hessian_sampled,
            LossHessianStrategy.AVERAGE: self.sum_hessian,
        }

    def backpropagate(self, module, grad_input, grad_output):
        curv_type = self._get_curv_type_from_extension()
        Curvature.check_loss_hessian(
            self.hessian_is_psd(), curv_type=curv_type)

        hessian_strategy = self._get_hessian_strategy_from_extension()
        H_func = self.LOSS_HESSIAN_GETTERS[hessian_strategy]
        H_loss = H_func(module, grad_input, grad_output)

        self.set_mat_in_ctx(H_loss)

    def _get_hessian_strategy_from_extension(self):
        strategy = self._get_parametrized_ext().get_loss_hessian_strategy()
        LossHessianStrategy.check_exists(strategy)
        return strategy


class HBPMSELoss(HBPLoss, MSELossDerivatives):
    pass


class HBPCrossEntropyLoss(HBPLoss, CrossEntropyLossDerivatives):
    pass


EXTENSIONS = [HBPCrossEntropyLoss(), HBPMSELoss()]
