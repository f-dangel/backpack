"""Utility functions."""

import opt_einsum as oe
import torch

TORCH = "torch"
OPT_EINSUM = "opt_einsum"

BPEXTS_EINSUM = "torch"


def _oe_einsum(equation, *operands):
    # handle old interface, passing operands as one list
    # see https://pytorch.org/docs/stable/_modules/torch/functional.html#einsum
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = operands[0]
    return oe.contract(equation, *operands, backend='torch')


EINSUMS = {
    TORCH: torch.einsum,
    OPT_EINSUM: _oe_einsum,
}


def einsum(equation, *operands):
    """`einsum` implementations used by `backpack`.

    Modify by setting `backpack.utils.utils.BPEXTS_EINSUM`.
    See `backpack.utils.utils.EINSUMS` for supported implementations.
    """
    return EINSUMS[BPEXTS_EINSUM](equation, *operands)


def random_psd_matrix(dim, device=None):
    """Random positive semi-definite matrix on device."""
    if device is None:
        device = torch.device("cpu")

    rand_mat = torch.randn(dim, dim, device=device)
    rand_mat = 0.5 * (rand_mat + rand_mat.t())
    shift = dim * torch.eye(dim, device=device)
    return rand_mat + shift
