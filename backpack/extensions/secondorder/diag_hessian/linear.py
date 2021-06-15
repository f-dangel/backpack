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
            h_diag_curr = LinUtils.extract_bias_diagonal(module, h_sqrt, sum_batch=True)
            h_diag.add_(sign * h_diag_curr)
        return h_diag

    def weight(self, ext, module, g_inp, g_out, backproped):
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]
        h_diag = torch.zeros_like(module.weight)

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag_curr = LinUtils.extract_weight_diagonal(
                module, h_sqrt, sum_batch=True
            )
            h_diag.add_(sign * h_diag_curr)
        return h_diag


class BatchDiagHLinear(DiagHBaseModule):
    """
    Individual Diagonal of the Hessian for torch.nn.linear

    Returns: Diagonal Hessian of weights
             Diagonal Hessian of bias
    """

    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, backproped):
        """
        Individual Diagonal of the Hessian for bias of torch.nn.linear
        """
        N = module.input0.shape[0]
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]
        h_diag = torch.zeros(
            N, *module.bias.shape, device=module.bias.device, dtype=module.bias.dtype
        )

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag_curr = LinUtils.extract_bias_diagonal(
                module, h_sqrt, sum_batch=False
            )
            h_diag.add_(sign * h_diag_curr)
        return h_diag

    def weight(self, ext, module, g_inp, g_out, backproped):
        """
        Individual Diagonal of the Hessian for weights of torch.nn.linear
        """
        N = module.input0.shape[0]
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]
        h_diag = torch.zeros(
            N,
            *module.weight.shape,
            device=module.weight.device,
            dtype=module.weight.dtype
        )

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag_curr = LinUtils.extract_weight_diagonal(
                module, h_sqrt, sum_batch=False
            )
            h_diag.add_(sign * h_diag_curr)
        return h_diag
