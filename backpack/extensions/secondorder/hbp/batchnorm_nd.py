from typing import Tuple, Union

from torch import Tensor, einsum
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.secondorder.hbp.hbpbase import HBPBaseModule
from backpack.utils.errors import batch_norm_raise_error_if_train


class HBPBatchNormNd(HBPBaseModule):
    def __init__(self):
        super().__init__(BatchNormNdDerivatives(), params=["weight", "bias"])

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        x_hat, _ = self.derivatives._get_normalized_input_and_var(module)
        v = backproped
        JTv = einsum("mnc...,nc...->mnc", v, x_hat)
        kfac_gamma = einsum("mnc...,mnd...->cd", JTv, JTv)
        return [kfac_gamma]

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        v = backproped
        JTv = v
        kfac_beta = einsum("mnc...,mnd...->cd", JTv, JTv)
        return [kfac_beta]

    def check_hyperparameters_module_extension(
        self,
        ext: BackpropExtension,
        module: Union[BatchNorm1d, BatchNorm2d, BatchNorm3d],
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
    ) -> None:  # noqa: D102
        batch_norm_raise_error_if_train(module)
