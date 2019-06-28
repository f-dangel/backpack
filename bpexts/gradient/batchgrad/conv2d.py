import torch.nn
from torch import einsum


def grad_batch(module, grad_input, grad_output):
    """Compute individual gradients for module parameters.

    Store bias batch gradients in `module.bias.batch_grad` and
    weight batch gradients in `module.weight.batch_grad`.
    """
    if module.bias is not None and module.bias.requires_grad:
        module.bias.grad_batch = bias_grad_batch(module, grad_output)
    if module.weight.requires_grad:
        module.weight.grad_batch = weight_grad_batch(module, grad_output)


def bias_grad_batch(module, grad_output):
    """Compute bias batch gradients.

    The batchwise gradient of a linear layer is simply given
    by the gradient with respect to the layer's output, summed over
    the spatial dimensions (each number in the bias vector is added.
    to the spatial output of an entire channel).
    """
    return grad_output[0].sum(3).sum(2)


def unfold_func(module):
    return torch.nn.Unfold(
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride
    )


def weight_grad_batch(module, grad_output):
    """Compute weight batch gradients.

    The linear layer applies
    Y = W * X
    (neglecting the bias) to a single sample X, where W denotes the
    matrix view of the kernel, Y is a view of the output and X denotes
    the unfolded input matrix.

    Note on shapes/dims:
    --------------------
    original input x: (batch_size, in_channels, x_dim, y_dim)
    original kernel w: (out_channels, in_channels, k_x, k_y)
    im2col input X: (batch_size, num_patches, in_channels * k_x * k_y)
    kernel matrix W: (out_channels, in_channels *  k_x * k_y)
    matmul result Y: (batch_size, out_channels, num_patches)
                   = (batch_size, out_channels, x_out_dim * y_out_dim)
    col2im output y: (batch_size, out_channels, x_out_dim, y_out_dim)

    Forward pass: (pseudo)
    -------------
    X = unfold(x) = im2col(x)
    W = view(w)
    Y[b,i,j] = W[i,m] *  X[b,m,j]
    y = view(Y)

    Backward pass: (pseudo)
    --------------
    Given: dE/dy    (same shape as y)
    dE/dY = view(dE/dy) (same shape as Y)

    dE/dW[b,k,l]    (batch-wise gradient)
    dE/dW[b,k,l] = (dY[b,i,j]/dW[k,l]) * dE/dY[b,i,j]
                 = delta(i,k) * delta(m,l) * X[b,m,j] * dE/dY[b,i,j]
                 = delta(m,l) * X[b,m,j] * dE/dY[b,k,j]

    Result:
    -------
    dE/dw = view(dE/dW)
    """
    batch_size = grad_output[0].size(0)
    dE_dw_shape = (batch_size, ) + module.weight.size()
    X = unfold_func(module)(module.input0)
    dE_dY = grad_output[0].view(batch_size, module.out_channels, -1)
    dE_dW = einsum('bml,bkl->bmk', (dE_dY, X))
    return dE_dW.view(dE_dw_shape)


SIGNATURE = [(torch.nn.Conv2d, "BATCH_GRAD", grad_batch)]
