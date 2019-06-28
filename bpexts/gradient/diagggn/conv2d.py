import torch.nn
from torch import einsum
from ..config import CTX
from ..batchgrad.conv2d import unfold_func
from ..jmp.conv2d import jac_mat_prod


def diag_ggn(module, grad_input, grad_output):
    sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
    if module.bias is not None and module.bias.requires_grad:
        module.bias.diag_ggn = bias_diag_ggn(module, grad_output, sqrt_ggn_out)
    if module.weight.requires_grad:
        module.weight.diag_ggn = weight_diag_ggn(module, grad_output,
                                                 sqrt_ggn_out)

    backpropagate_sqrt_ggn(module, grad_input, grad_output, sqrt_ggn_out)


def bias_diag_ggn(module, grad_output, sqrt_ggn_out):
    # separate channel and spatial coordinates of features
    batch = module.input0.size(0)
    num_classes = sqrt_ggn_out.size(2)
    out_pixels = module.output_shape[2] * module.output_shape[3]
    # apply Jacobian
    sqrt_ggn = sqrt_ggn_out.view(batch, module.out_channels, out_pixels,
                                 num_classes)
    sqrt_ggn = einsum('bijc->bic', sqrt_ggn)
    # extract diag
    return einsum('bic->i', (sqrt_ggn**2))


# TODO: Move the axis-merging trick to a separate method
def weight_diag_ggn(module, grad_output, sqrt_ggn_out):
    # checks for debugging
    batch, out_channels, out_x, out_y = module.output_shape
    in_features = module.input0.numel() / batch
    out_features = out_channels * out_x * out_y
    num_classes = sqrt_ggn_out.size(2)
    assert tuple(sqrt_ggn_out.size())[:2] == (batch, out_features)

    # separate channel and spatial dimensions
    sqrt_ggn = sqrt_ggn_out.view(batch, out_channels, out_x * out_y,
                                 num_classes)
    # unfolded input, repeated for each class
    X = unfold_func(module)(module.input0)
    X = X.unsqueeze(0).expand(num_classes, -1, -1, -1)

    # apply Jacobian
    sqrt_ggn = einsum('bmlc,cbkl->bmkc', (sqrt_ggn, X)).contiguous()
    # compute the diagonal
    w_diag_ggn = einsum('bmkc->mk', (sqrt_ggn**2, ))
    # reshape into kernel dimensions
    return w_diag_ggn.view_as(module.weight)


def backpropagate_sqrt_ggn(module, grad_input, grad_output, sqrt_ggn_out):
    sqrt_ggn_in = jac_mat_prod(module, grad_input, grad_output, sqrt_ggn_out)
    CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


SIGNATURE = [(torch.nn.Conv2d, "DIAG_GGN", diag_ggn)]
