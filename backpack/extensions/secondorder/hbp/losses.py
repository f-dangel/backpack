from functools import partial

from backpack.core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from backpack.core.derivatives.mseloss import MSELossDerivatives
from backpack.extensions.curvature import Curvature
from backpack.extensions.secondorder.hbp.hbp_options import LossHessianStrategy
from backpack.extensions.secondorder.hbp.hbpbase import HBPBaseModule


class HBPLoss(HBPBaseModule):
    def backpropagate(self, ext, module, g_inp, g_out, backproped):
        Curvature.check_loss_hessian(
            self.derivatives.hessian_is_psd(), curv_type=ext.get_curv_type()
        )

        H_func = self.make_loss_hessian_func(ext)
        H_loss = H_func(module, g_inp, g_out)

        return H_loss

    def make_loss_hessian_func(self, ext):
        """Get function that produces the backpropagated quantity."""
        hessian_strategy = ext.get_loss_hessian_strategy()

        if hessian_strategy == LossHessianStrategy.EXACT:
            return self.derivatives.sqrt_hessian
        elif hessian_strategy == LossHessianStrategy.SAMPLING:
            mc_samples = ext.get_num_mc_samples()
            return partial(self.derivatives.sqrt_hessian_sampled, mc_samples=mc_samples)
        elif hessian_strategy == LossHessianStrategy.SUM:
            return self.derivatives.sum_hessian
        else:
            raise ValueError("Unknown Hessian strategy: {}".format(hessian_strategy))


class HBPMSELoss(HBPLoss):
    def __init__(self):
        super().__init__(derivatives=MSELossDerivatives())


class HBPCrossEntropyLoss(HBPLoss):
    def __init__(self):
        super().__init__(derivatives=CrossEntropyLossDerivatives())
