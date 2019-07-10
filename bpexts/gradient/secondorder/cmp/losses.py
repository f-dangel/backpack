from ...derivatives.mseloss import MSELossDerivatives
from ...derivatives.crossentropyloss import CrossEntropyLossDerivatives
from .cmpbase import CMPBase, HESSIAN, PCH_CLIP, PCH_ABS


class CMPLoss(CMPBase):
    def backpropagate(self, module, grad_input, grad_output):
        hmp = self.hessian_matrix_product(module, grad_input, grad_output)

        def CMP(mat, which=HESSIAN):
            self._check_loss_hessian_usable_for_curvature(which)
            return hmp(mat)

        self.set_in_ctx(CMP)

    def _check_loss_hessian_usable_for_curvature(self, which):
        require_psd = self.REQUIRE_PSD_LOSS_HESSIAN[which]
        is_psd = self.hessian_is_psd()
        if require_psd and not is_psd:
            raise ValueError(
                'Loss Hessian PSD = {}, but {} requires PSD = {}'.format(
                    is_psd, which, require_psd))


class CMPMSELoss(CMPLoss, MSELossDerivatives):
    pass


class CMPCrossEntropyLoss(CMPLoss, CrossEntropyLossDerivatives):
    pass


EXTENSIONS = [CMPCrossEntropyLoss(), CMPMSELoss()]
