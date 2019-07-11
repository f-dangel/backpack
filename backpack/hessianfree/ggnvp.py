import torch
from ..utils import einsum

from .lop import L_op
from .rop import R_op
from .hvp import hessian_vector_product


def ggn_vector_product(loss, output, model, vp):
    """Compute GGN-vector product G loss(x) * v."""
    plist = list(model.parameters())
    return ggn_vector_product_from_plist(loss, output, plist, vp)


def ggn_vector_product_from_plist(loss, output, plist, vp):
    Jv = R_op(output, plist, vp)
    batch, dims = output.size(0), output.size(1)
    # TODO: Clean up
    if loss.grad_fn.__class__.__name__ == 'NllLossBackward':
        outputsoftmax = torch.nn.functional.softmax(output, dim=1)
        M = torch.zeros(batch, dims,
                        dims).cuda() if outputsoftmax.is_cuda else torch.zeros(
                            batch, dims, dims)
        M.reshape(batch, -1)[:, ::dims + 1] = outputsoftmax
        H = M - einsum('bi,bj->bij', (outputsoftmax, outputsoftmax))
        HJv = [torch.squeeze(H @ torch.unsqueeze(Jv[0], -1)) / batch]
    else:
        HJv = hessian_vector_product(loss, output, Jv)
    JHJv = L_op(output, plist, HJv)
    return JHJv
