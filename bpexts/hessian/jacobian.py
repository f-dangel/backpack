"""Exact computation of the full Jacobian."""

from torch import (cat, zeros)
from torch.autograd import grad
from tqdm import tqdm


def exact_jacobian(f, parameters, show_progress=True):
    """Compute the excact Jacobian matrix.

    Parameters:
    -----------
    f (tensor): Tensor-valued function.
    parameters (list/tuple/iterator): Iterable object containing all
               tensor-valued argument of `f`.
    show_progress (bool): Show a progressbar which also estimates the
                  remaining runtime

    The generalized Jacobian for a tensor-valued function `f` of
    a tensor-valued input `x` is given by d vec(f)/ d vec(x), where
    the vectorization corresponds to a one-dimensional view of the
    tensor.

    Returns matrix with `J[i,j] = d vec(f)[i] / d vec(x)[j]`.
    """
    params = list(parameters)
    num_params = int(sum(p.numel() for p in params))
    if not all(p.requires_grad for p in params):
        raise ValueError('All parameters have to require_grad')
    jacobian = zeros(f.numel(), num_params)
    progressbar = tqdm(iterable=range(f.numel()),
                       total=f.numel(),
                       desc='[exact] Jacobian',
                       disable=(not show_progress))
    f_flat = f.view(-1)
    for idx in progressbar:
        df = grad(f_flat[idx], parameters, create_graph=True)
        dtheta = None
        for d in df:
            dtheta = d.contiguous().view(-1) if dtheta is None\
                     else cat([dtheta, d.contiguous().view(-1)])
        jacobian[idx] = dtheta
    return jacobian
