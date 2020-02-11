from functools import partial

from backpack.core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from backpack.core.derivatives.mseloss import MSELossDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule
from backpack.extensions.secondorder.hbp import LossHessianStrategy


class DiagGGNLoss(DiagGGNBaseModule):
    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):
        hess_func = self.make_loss_hessian_func(ext)

        return hess_func(module, grad_inp, grad_out)

    def make_loss_hessian_func(self, ext):
        """Get function that produces the backpropagated quantity."""
        loss_hessian_strategy = ext.loss_hessian_strategy

        if loss_hessian_strategy == LossHessianStrategy.EXACT:
            return self.derivatives.sqrt_hessian
        elif loss_hessian_strategy == LossHessianStrategy.SAMPLING:
            mc_samples = ext.get_num_mc_samples()
            return partial(self.derivatives.sqrt_hessian_sampled, mc_samples=mc_samples)

        else:
            raise ValueError(
                "Unknown hessian strategy {}".format(loss_hessian_strategy)
            )


class DiagGGNMSELoss(DiagGGNLoss):
    def __init__(self):
        super().__init__(derivatives=MSELossDerivatives())


class DiagGGNCrossEntropyLoss(DiagGGNLoss):
    def __init__(self):
        super().__init__(derivatives=CrossEntropyLossDerivatives())
