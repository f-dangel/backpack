from ...derivatives.mseloss import MSELossDerivatives
from ...derivatives.crossentropyloss import CrossEntropyLossDerivatives
from .diagggnbase import DiagGGNBase


class DiagGGNLoss(DiagGGNBase):
    def backpropagate(self, module, grad_input, grad_output):
        sqrt_H = self.sqrt_hessian(module, grad_input, grad_output)
        self.set_in_ctx(sqrt_H)


class DiagGGNMSELoss(DiagGGNLoss, MSELossDerivatives):
    pass


class DiagGGNCrossEntropyLoss(DiagGGNLoss, CrossEntropyLossDerivatives):
    pass


EXTENSIONS = [DiagGGNCrossEntropyLoss(), DiagGGNMSELoss()]
