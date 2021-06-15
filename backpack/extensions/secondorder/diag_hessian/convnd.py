import torch
import torch.nn

from backpack.extensions.secondorder.diag_hessian.diag_h_base import DiagHBaseModule
from backpack.utils import conv as convUtils


class DiagHConvND(DiagHBaseModule):
    def bias(self, ext, module, g_inp, g_out, backproped):
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]
        h_diag = torch.zeros_like(module.bias)

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag.add_(
                convUtils.extract_bias_diagonal(module, h_sqrt, sum_batch=True),
                alpha=sign,
            )

        return h_diag

    def weight(self, ext, module, g_inp, g_out, backproped):
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]
        X = convUtils.unfold_input(module, module.input0)
        h_diag = torch.zeros_like(module.weight)

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag.add_(
                convUtils.extract_weight_diagonal(module, X, h_sqrt, sum_batch=True),
                alpha=sign,
            )

        return h_diag


class BatchDiagHConvND(DiagHBaseModule):
    def bias(self, ext, module, g_inp, g_out, backproped):
        N = module.input0.shape[0]
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]
        h_diag = torch.zeros(
            N, *module.bias.shape, device=module.bias.device, dtype=module.bias.dtype
        )

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag.add_(
                convUtils.extract_bias_diagonal(module, h_sqrt, sum_batch=False),
                alpha=sign,
            )

        return h_diag

    def weight(self, ext, module, g_inp, g_out, backproped):
        N = module.input0.shape[0]
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]
        X = convUtils.unfold_input(module, module.input0)
        h_diag = torch.zeros(
            N,
            *module.weight.shape,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag.add_(
                convUtils.extract_weight_diagonal(module, X, h_sqrt, sum_batch=False),
                alpha=sign,
            )

        return h_diag
