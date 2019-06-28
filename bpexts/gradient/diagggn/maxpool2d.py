import torch.nn
from torch import einsum
from ..config import CTX


def diag_ggn(module, grad_output):
    sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
    CTX._backpropagated_sqrt_ggn = backpropagate_sqrt_ggn(
        module, grad_output, sqrt_ggn_out)


def backpropagate_sqrt_ggn(module, grad_output, sqrt_ggn_out):
    # TODO: Need access to indices, can be obtained from forward pass
    _, pool_idx = torch.nn.functional.max_pool2d(
        module.input0,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        return_indices=True,
        ceil_mode=module.ceil_mode)

    # checks for debugging
    batch, channels, out_x, out_y = module.output_shape
    _, _, in_x, in_y = module.input0.size()
    in_features = module.input0.numel() / batch
    out_features = channels * out_x * out_y
    num_classes = sqrt_ggn_out.size(2)
    assert tuple(sqrt_ggn_out.size())[:2] == (batch, out_features)

    ##############################################################
    # trick to process batch and class dimensions without for loop
    # shape of sqrt_ggn_out: (batch, output, class)
    # merge batch and class axes (einsum for convenience)
    # TODO: This can be formulated more efficiently
    sqrt_ggn = einsum('boc->cbo', (sqrt_ggn_out, )).contiguous()
    sqrt_ggn = sqrt_ggn.view(num_classes * batch, out_features)

    # separate channel and spatial dimensions
    sqrt_ggn = sqrt_ggn.view(num_classes * batch, channels, out_x * out_y)
    # end TODO

    # apply Jacobian w.r.t. input
    result = torch.zeros(
        num_classes * batch, channels, in_x * in_y, device=sqrt_ggn.device)
    pool_idx = pool_idx.view(batch, channels, out_x * out_y)
    pool_idx = pool_idx.repeat(num_classes, 1, 1)
    result.scatter_add_(2, pool_idx, sqrt_ggn)

    # unmerge batch and class axes
    result = result.view(num_classes, batch, in_features)
    return einsum('cbi->bic', (result, ))


SIGNATURE = [(torch.nn.MaxPool2d, "DIAG_GGN", diag_ggn)]
