import torch

import backpack.utils.linear as LinUtils
from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.secondorder.diag_hessian.diag_h_base import DiagHBaseModule


class DiagHLinear(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, backproped):
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]
        h_diag = torch.zeros_like(module.bias)

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag.add_(
                LinUtils.extract_bias_diagonal(module, h_sqrt, sum_batch=True),
                alpha=sign,
            )

        return h_diag

    def weight(self, ext, module, g_inp, g_out, backproped):
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]
        h_diag = torch.zeros_like(module.weight)

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag.add_(
                LinUtils.extract_weight_diagonal(module, h_sqrt, sum_batch=True),
                alpha=sign,
            )

        return h_diag


class BatchDiagHLinear(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, backproped):
        N = module.input0.shape[0]
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]
        h_diag = torch.zeros(
            N, *module.bias.shape, device=module.bias.device, dtype=module.bias.dtype
        )

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag.add_(
                LinUtils.extract_bias_diagonal(module, h_sqrt, sum_batch=False),
                alpha=sign,
            )

        return h_diag

    def weight(self, ext, module, g_inp, g_out, backproped):
        N = module.input0.shape[0]
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]
        h_diag = torch.zeros(
            N,
            *module.weight.shape,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag.add_(
                LinUtils.extract_weight_diagonal(module, h_sqrt, sum_batch=False),
                alpha=sign,
            )
        return h_diag
