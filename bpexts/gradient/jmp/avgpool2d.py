"""The code relies on the insight that average pooling can be understood as
convolution over single channels with a constant kernel."""

import torch.nn
from torch.nn.functional import conv_transpose2d
from .maxpool2d import check_sizes_input, reshape_for_matmul
from .conv2d import jac_mat_prod
from ...utils import einsum

# NOTE: Might now work for exotic hyperparameters of AvgPool2d,
# like count_include_pad=False


def jac_mat_prod(module, grad_input, grad_output, mat):
    check_sizes_input(mat, module)
    mat_as_pool = reshape_for_conv(mat, module)
    jmp_as_pool = apply_jacobian_of(module, mat_as_pool)

    batch, channels, in_x, in_y = module.input0.size()
    num_classes = mat.size(2)
    assert jmp_as_pool.size(0) == num_classes * batch * channels
    assert jmp_as_pool.size(1) == 1
    assert jmp_as_pool.size(2) == in_x
    assert jmp_as_pool.size(3) == in_y

    return reshape_for_matmul(jmp_as_pool, module)


def reshape_for_conv(mat, module):
    """Create fake single-channel images, grouping batch,
    class and channel dimension."""
    batch, channels, out_x, out_y = module.output_shape
    num_classes = mat.size(-1)

    # 'fake' image for convolution
    # (batch * class * channel, 1,  out_x, out_y)
    return einsum('bic->bci', mat).contiguous().view(
        batch * num_classes * channels, 1, out_x, out_y)


def reshape_for_matmul(mat, module):
    """Ungroup dimensions after application of Jacobian."""
    batch, channels, in_x, in_y = module.input0.size()
    features = channels * in_x * in_y
    # mat is of shape (batch * class * channel, 1,  in_x, in_y)
    # move class dimension to last
    mat_view = mat.view(batch, -1, features)
    return einsum('bci->bic', mat_view).contiguous()


def apply_jacobian_of(module, mat):
    _, _, in_x, in_y = module.input0.size()
    output_size = (mat.size(0), 1, in_x, in_y)

    # need to use Conv2dTranspose to fix dimension of output
    conv2d_t = torch.nn.ConvTranspose2d(
        1,
        1,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        bias=False).to(module.input0.device)
    avg_kernel = torch.ones_like(conv2d_t.weight) / conv2d_t.weight.numel()
    conv2d_t.weight.data = avg_kernel

    with torch.no_grad():
        result = conv2d_t(mat, output_size=output_size)
    return result
