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

    # separate channel and spatial dimensions
    sqrt_ggn = sqrt_ggn_out.view(batch, channels, out_x * out_y, num_classes)
    # apply Jacobian w.r.t. input
    result = torch.zeros(
        batch, channels, in_x * in_y, num_classes, device=sqrt_ggn.device)
    pool_idx = pool_idx.view(batch, channels, out_x * out_y)
    pool_idx = pool_idx.unsqueeze(-1).expand(-1, -1, -1, num_classes)
    result.scatter_add_(2, pool_idx, sqrt_ggn)
    return result.view(batch, in_features, num_classes)


SIGNATURE = [(torch.nn.MaxPool2d, "DIAG_GGN", diag_ggn)]
