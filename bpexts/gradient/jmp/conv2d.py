from torch import einsum
from torch.nn.functional import conv_transpose2d


def jac_mat_prod(module, grad_input, grad_output, mat):

    check_sizes_input(mat, module)
    mat_as_conv = reshape_for_conv(mat, module)
    jmp_as_conv = apply_jacobian_of(module, mat_as_conv)
    check_sizes_output(jmp_as_conv, module)

    return reshape_for_matmul(jmp_as_conv, module)


def check_sizes_input(mat, module):
    batch, out_channels, out_x, out_y = module.output_shape
    assert tuple(mat.size())[:2] == (batch, out_channels * out_x * out_y)


def check_sizes_output(jmp, module):
    if tuple(jmp.size())[1:] != tuple(module.input0.size())[1:]:
        raise ValueError(
            "Size after conv_transpose does not match",
            "Got {}, and {}.",
            "Expected all dimensions to match, except for the first."
            .format(jmp.size(), module.input0.size())
        )


def reshape_for_conv(bmat, module):
    batch, out_channels, out_x, out_y = module.output_shape
    num_classes = bmat.size(2)

    bmat = einsum('boc->cbo', (bmat, )).contiguous()
    bmat = bmat.view(num_classes * batch, out_channels * out_x * out_y)
    bmat = bmat.view(num_classes * batch, out_channels, out_x, out_y)
    return bmat


def reshape_for_matmul(bconv, module):
    batch, _, _, _ = module.output_shape
    in_features = module.input0.numel() / batch
    bconv = bconv.view(-1, batch, in_features)
    bconv = einsum('cbi->bic', (bconv, ))
    return bconv


def apply_jacobian_of(module, mat):
    return conv_transpose2d(
        mat,
        module.weight.data,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups
    )
