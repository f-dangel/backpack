import torch
import torch.nn

from backpack.extensions.secondorder.diag_hessian.diag_h_base import DiagHBaseModule
from backpack.utils import conv as convUtils


class DiagHConvND(DiagHBaseModule):
    def __init__(self, derivatives, N, params=None):
        super().__init__(derivatives=derivatives, params=["bias", "weight"])
        self.N = N

    def bias(self, ext, module, g_inp, g_out, backproped):
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]
        h_diag = torch.zeros_like(module.bias)

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag_curr = convUtils.extract_bias_diagonal(
                module, h_sqrt, self.N, sum_batch=True
            )
            h_diag.add_(sign * h_diag_curr)
        return h_diag

    def weight(self, ext, module, g_inp, g_out, backproped):
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]
        X = convUtils.unfold_by_conv(module.input0, module)
        h_diag = torch.zeros_like(module.weight)

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag_curr = convUtils.extract_weight_diagonal(
                module, X, h_sqrt, self.N, sum_batch=True
            )
            h_diag.add_(sign * h_diag_curr)
        return h_diag


class BatchDiagHConvND(DiagHBaseModule):
    '''
    Individual Diagonal of the Hessian for torch.nn.ConvNd

    Returns: Diagonal Hessian of weights
             Diagonal Hessian of bias
    '''
    def __init__(self, derivatives, N, params=None):
        super().__init__(derivatives=derivatives, params=["bias", "weight"])
        self.N = N

    def bias(self, ext, module, g_inp, g_out, backproped):
        '''
        Individual Diagonal of the Hessian for bias of torch.nn.ConvNd
        '''
        N = module.input0.shape[0]
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]
        h_diag = torch.zeros(N, *module.bias.shape)

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag_curr = convUtils.extract_bias_diagonal(
                module, h_sqrt, self.N, sum_batch=False
            )
            h_diag.add_(sign * h_diag_curr)
        return h_diag

    def weight(self, ext, module, g_inp, g_out, backproped):
        '''
        Individual Diagonal of the Hessian for weights of torch.nn.ConvNd
        '''
        N = module.input0.shape[0]
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]
        X = convUtils.unfold_by_conv(module.input0, module)
        h_diag = torch.zeros(N, *module.weight.shape)

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag_curr = convUtils.extract_weight_diagonal(
                module, X, h_sqrt, self.N, sum_batch=False
            )
            h_diag.add_(sign * h_diag_curr)
        return h_diag
