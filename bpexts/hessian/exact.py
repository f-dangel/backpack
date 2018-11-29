"""Exact computation of full Hessian."""

from torch import (cat, zeros)
from torch.autograd import grad
from tqdm import tqdm


def exact_hessian(f, parameters, show_progress=True):
    """Compute all second derivatives of a scalar w.r.t. `parameters`.

    The order of parameters corresponds to a one-dimensional
     vectorization followed by a concatenation of all tensors in
    `parameters`.

    Parameters:
    -----------
    f (scalar): Scalar PyTorch function/tensor.
    parameters (list/tuple/iterator): iterable object containing all
                tensors acting as variables of `f`
    show_progress (bool): Show a progressbar which also estimates the
                          remaining runtime

    Details:
    --------
    The parameters in the list are all flattened and concatenated
    into one large vector `theta`. Return the matrix `d^2 E/ d^2 theta`
    with `(d^2E / d^2 theta)[i, j]` =  (d^2E / d theta[i] d theta[j]).

    Taken from
        https://discuss.pytorch.org/t/compute-the-hessian-matrix-of-a-
        network/15270/3
    """
    params = list(parameters)
    if not all(p.requires_grad for p in params):
        raise ValueError('All parameters have to require_grad')
    df = grad(f, params, create_graph=True)
    # flatten all parameter gradients and concatenate into a vector
    dtheta = None
    for grad_f in df:
        dtheta = grad_f.contiguous().view(-1) if dtheta is None\
                else cat([dtheta, grad_f.contiguous().view(-1)])
    # compute second derivatives
    hessian_dim = dtheta.size(0)
    hessian = zeros(hessian_dim, hessian_dim)
    progressbar = tqdm(iterable=range(hessian_dim),
                       total=hessian_dim,
                       desc='[exact] Full Hessian',
                       disable=(not show_progress))
    for idx in progressbar:
        df2 = grad(dtheta[idx], params, create_graph=True)
        d2theta = None
        for d2 in df2:
            d2theta = d2.contiguous().view(-1) if d2theta is None\
                      else cat([d2theta, d2.contiguous().view(-1)])
        hessian[idx] = d2theta
    return hessian
