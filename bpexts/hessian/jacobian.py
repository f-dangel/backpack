"""Exact computation of the full Jacobian."""

from torch import (cat, zeros)
from torch.autograd import grad
from tqdm import tqdm


def exact_jacobian(f, parameters, show_progress=False):
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
        df = grad(f_flat[idx], params, create_graph=True)
        dtheta = None
        for d in df:
            dtheta = d.contiguous().view(-1) if dtheta is None\
                     else cat([dtheta, d.contiguous().view(-1)])
        jacobian[idx] = dtheta
    return jacobian


def exact_jacobian_batchwise(f, parameter, show_progress=False):
    """Jacobian of batch function and parameter (inefficient).

    Both the 0th axis of `f` and `parameter` have to correspond
    to the batch dimension.

    Note:
    -----
    Take into account that the function and parameters were processed
    batch-wise. This leads to a sparse structure (i.e. block-diagonality)
    of the exact Jacobian.
    
    Given a tensor function `f` of shape `(batch_size, dim_x, dim_y, ...)`
    with a `parameter` of shape `(batch_size, dim_1, dim_2, ...)`,
    the batchwise Jacobian is a matrix of size 
    `(batch_size, dim_x * dim_y * ..., dim_1 * dim_2 * ...)`.

    When computing the Jacobian of a function whose output is known to
    consist of batch samples along the 0th axis with respect to batch-shaped
    parameters, it is sufficient to store the Jacobian for each
    sample/parameter pair. Speaking differently, the 0th axis of the
    computed batchwise Jacobian holds all exact Jacobians of the 0th axis
    of the function w.r.t. the zeroth axes of all parameters.   
    """
    batch_size = f.size()[0]
    if not parameter.size()[0] == batch_size:
        raise ValueError('Parameter does not have batch dimension of f')
    jacobian = exact_jacobian(f, [parameter], show_progress=show_progress)
    # reshape, grouping the batch dimensions of f and p
    no_batch_f = f.numel() // batch_size
    no_batch_p = parameter.numel() // batch_size
    jacobian = jacobian.view(batch_size, no_batch_f, batch_size, no_batch_p)
    # cut out diagonal blocks
    range_b = list(range(batch_size))
    return jacobian[range_b, :, range_b, :]
