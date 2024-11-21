"""Matrix-free multiplication with the Hessian."""

from typing import Optional, Sequence, Tuple, Union

from torch import Tensor
from torch.autograd import grad
from torch.nn import Parameter


def hessian_vector_product(
    f: Tensor,
    params: Sequence[Union[Tensor, Parameter]],
    v: Sequence[Tensor],
    grad_params: Optional[Sequence[Tensor]] = None,
    detach: bool = True,
) -> Tuple[Tensor, ...]:
    """Multiply a vector ``v`` with the Hessian of ``f`` w.r.t. ``params``.

    Args:
        f: A scalar-valued tensor whose Hessian is multiplied onto the vector.
        params: Parameters w.r.t. which the Hessian is computed.
        v: Vector that is multiplied with the Hessian. Entries must have same shape
            as the entries in ``params``.
        grad_params: Pre-computed gradients of ``f`` w.r.t. ``params``. Useful if the
            gradient is computed elsewhere. If provided, the first backward pass can be
            avoided. Gradients must have been computed with ``create_graph=True``.
            Entries must have same shape as the entries in ``params``.
        detach: Whether to detach the Hessian-vector product from the computation graph.

    Returns:
        Hessian-vector product of ``f`` with respect to ``params`` applied to ``v``.
        Entries have same shape as entries of ``params`` and ``v``.

    Raises:
        ValueError: If the length or shapes of ``params``, ``v``, and ``grad_params``
            do not match.
    """
    if grad_params is None:
        grad_params = grad(
            f, params, create_graph=True, retain_graph=True, materialize_grads=True
        )

    if not len(grad_params) == len(params) == len(v):
        raise ValueError(
            f"Expected {len(params)} parameters, gradients, and vectors, "
            f"but got {len(params)}, {len(grad_params)}, and {len(v)}."
        )
    if not all(
        p_i.shape == v_i.shape == g_i.shape
        for p_i, v_i, g_i in zip(params, v, grad_params)
    ):
        raise ValueError(
            "Expected parameters, vectors, and gradients to have the same shape, "
            f"but got {[p_i.shape for p_i in params]}, {[v_i.shape for v_i in v]}, "
            f"and {[g_i.shape for g_i in grad_params]}."
        )

    gv = sum((g_i * v_i).sum() for g_i, v_i in zip(grad_params, v))
    Hv = grad(gv, params, create_graph=True, retain_graph=True, materialize_grads=True)

    return tuple(j.detach() for j in Hv) if detach else Hv
