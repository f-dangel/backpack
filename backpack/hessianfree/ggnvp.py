"""Autodiff-only matrix-free multiplication by the generalized Gauss-Newton/Fisher."""
from typing import List, Tuple

from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter

from backpack.hessianfree.hvp import hessian_vector_product
from backpack.hessianfree.lop import L_op
from backpack.hessianfree.rop import R_op


def ggn_vector_product(
    loss: Tensor, output: Tensor, model: Module, v: List[Tensor]
) -> Tuple[Tensor]:
    """Multiply a vector with the generalized Gauss-Newton/Fisher.

    Note:
        ``G v = J.T @ H @ J @ v`` where ``J`` is the Jacobian of ``output`` w.r.t.
        ``model``'s trainable parameters and `H` is the Hessian of `loss` w.r.t.
        ``output``. ``v`` is the flattened and concatenated version of the passed
        list of vectors.

    Args:
        loss: Scalar tensor that represents the loss.
        output: Model output.
        model: The model.
        v: Vector specified as list of tensors matching the trainable parameters.

    Returns:
        GGN-vector product in list format, i.e. as list that matches the sizes
        of trainable model parameters.
    """
    return ggn_vector_product_from_plist(
        loss, output, [p for p in model.parameters() if p.requires_grad], v
    )


def ggn_vector_product_from_plist(
    loss: Tensor, output: Tensor, plist: List[Parameter], v: List[Tensor]
) -> Tuple[Tensor]:
    """Multiply a vector with a sub-block of the generalized Gauss-Newton/Fisher.

    Args:
        loss: Scalar tensor that represents the loss.
        output: Model output.
        plist: List of trainable parameters whose GGN block is used for multiplication.
        v: Vector specified as list of tensors matching the sizes of ``plist``.

    Returns:
        GGN-vector product in list format, i.e. as list that matches the sizes of
        ``plist``.
    """
    Jv = R_op(output, plist, v)
    HJv = hessian_vector_product(loss, output, Jv)
    JTHJv = L_op(output, plist, HJv)
    return JTHJv
