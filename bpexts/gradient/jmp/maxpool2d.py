import torch.nn
from torch.nn.functional import max_pool2d


def jac_mat_prod(module, grad_input, grad_output, mat):
    check_sizes_input(mat, module)
    mat_as_pool = reshape_for_pooling(mat, module)
    jmp_as_pool = apply_jacobian_of(module, mat_as_pool)
    return reshape_for_matmul(jmp_as_pool, module)


def check_sizes_input(mat, module):
    batch, out_channels, out_x, out_y = module.output_shape
    assert tuple(mat.size())[:2] == (batch, out_channels * out_x * out_y)


def reshape_for_pooling(mat, module):
    batch, channels, out_x, out_y = module.output_shape
    _, _, in_x, in_y = module.input0.size()
    num_classes = mat.size(-1)

    batch, channels, out_x, out_y = module.output_shape
    return mat.view(batch, channels, out_x * out_y, num_classes)


def reshape_for_matmul(mat, module):
    batch, channels, out_x, out_y = module.output_shape
    _, _, in_x, in_y = module.input0.size()
    in_features = module.input0.numel() / batch
    batch = mat.size(0)
    num_classes = mat.size(-1)

    return mat.view(batch, in_features, num_classes)


def apply_jacobian_of(module, mat):
    batch, channels, out_x, out_y = module.output_shape
    _, _, in_x, in_y = module.input0.size()
    num_classes = mat.shape[-1]

    result = torch.zeros(
        batch, channels, in_x * in_y, num_classes, device=mat.device
    )

    pool_idx = get_pooling_idx(module)
    pool_idx = pool_idx.view(batch, channels, out_x * out_y)
    pool_idx = pool_idx.unsqueeze(-1).expand(-1, -1, -1, num_classes)
    result.scatter_add_(2, pool_idx, mat)
    return result


def get_pooling_idx(module):
    _, pool_idx = max_pool2d(
        module.input0,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        return_indices=True,
        ceil_mode=module.ceil_mode
    )
    return pool_idx
