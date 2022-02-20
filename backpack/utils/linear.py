"""Contains utility functions to extract the GGN diagonal for linear layers."""
from torch import Tensor, einsum
from torch.nn import Linear


def extract_weight_diagonal(
    module: Linear, S: Tensor, sum_batch: bool = True
) -> Tensor:
    """Extract diagonal of ``(Jᵀ S) (Jᵀ S)ᵀ`` where ``J`` is the weight Jacobian.

    Args:
        module: Linear layer for which the diagonal is extracted w.r.t. the weight.
        S: Backpropagated symmetric factorization of the loss Hessian. Has shape
            ``(V, *module.output.shape)``.
        sum_batch: Sum out the weight diagonal's batch dimension. Default: ``True``.

    Returns:
        Per-sample weight diagonal if ``sum_batch=False`` (shape
        ``(N, module.weight.shape)`` with batch size ``N``) or summed weight diagonal
        if ``sum_batch=True`` (shape ``module.weight.shape``).
    """
    has_additional_axes = module.input0.dim() > 2

    if has_additional_axes:
        S_flat = S.flatten(start_dim=2, end_dim=-2)
        X_flat = module.input0.flatten(start_dim=1, end_dim=-2)
        equation = f"vnmo,nmi,vnko,nki->{'' if sum_batch else 'n'}oi"
        # TODO Compare `torch.einsum`, `opt_einsum.contract` and the base class
        # implementation: https://github.com/fKunstner/backpack-discuss/issues/111
        return einsum(equation, S_flat, X_flat, S_flat, X_flat)

    else:
        equation = f"vno,ni->{'' if sum_batch else 'n'}oi"
        return einsum(equation, S**2, module.input0**2)


# TODO This method applies the bias Jacobian, then squares and sums the result. Intro-
# duce base class for {Batch}DiagHessian and DiagGGN{Exact,MC} and remove this method
def extract_bias_diagonal(module: Linear, S: Tensor, sum_batch: bool = True) -> Tensor:
    """Extract diagonal of ``(Jᵀ S) (Jᵀ S)ᵀ`` where ``J`` is the bias Jacobian.

    Args:
        module: Linear layer for which the diagonal is extracted w.r.t. the bias.
        S: Backpropagated symmetric factorization of the loss Hessian. Has shape
            ``(V, *module.output.shape)``.
        sum_batch: Sum out the bias diagonal's batch dimension. Default: ``True``.

    Returns:
        Per-sample bias diagonal if ``sum_batch=False`` (shape
        ``(N, module.bias.shape)`` with batch size ``N``) or summed bias diagonal
        if ``sum_batch=True`` (shape ``module.bias.shape``).
    """
    additional_axes = list(range(2, module.input0.dim()))

    if additional_axes:
        JS = S.sum(additional_axes)
    else:
        JS = S

    equation = f"vno->{'' if sum_batch else 'n'}o"

    return einsum(equation, JS**2)
