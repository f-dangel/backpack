import torch.nn
from torch import einsum


def grad_batch(module, grad_output):
    """Compute batchwise gradients of module parameters.

    Store bias batch gradients in `module.bias.batch_grad` and
    weight batch gradients in `module.weight.batch_grad`.
    """
    if module.bias is not None and module.bias.requires_grad:
        module.bias.grad_batch = bias_grad_batch(module, grad_output)
    if module.weight.requires_grad:
        module.weight.grad_batch = weight_grad_batch(module, grad_output)


def bias_grad_batch(module, grad_output):
    """Compute bias batch gradients from grad w.r.t. layer outputs.

    The batchwise gradient of a linear layer is simply given
    by the gradient with respect to the layer's output.
    """
    return grad_output[0]


def weight_grad_batch(module, grad_output):
    r"""Compute weight batch gradients from grad w.r.t layer outputs.

    The linear layer applies
    y = W x + b.
    to a single sample x, where W denotes the weight and b the bias.

    Result:
    -------
    Finally, this yields
    dE / d vec(W) = (dy / d vec(W))^T dy  (vec denotes row stacking)
                  = dy \otimes x^T

    Derivation:
    -----------
    The Jacobian for W is given by
    (matrix derivative notation)
    dy / d vec(W) = x^T \otimes I    (vec denotes column stacking),
    dy / d vec(W) = I \otimes x      (vec denotes row stacking),
    (dy / d vec(W))^T = I \otimes x^T  (vec denotes row stacking)
    or
    (index notation)
    dy[i] / dW[j,k] = delta(i,j) x[k]    (index notation).
    """
    batch_size = grad_output[0].size(0)
    weight_grad_batch = einsum('bi,bj->bij', (grad_output[0], module.input0))
    return weight_grad_batch.view(batch_size, module.out_features, module.in_features)


SIGNATURE = [(torch.nn.Linear, "BATCH_GRAD", grad_batch)]
