"""Exact computation of the full Jacobian."""

from torch import (cat, zeros)
from torch.autograd import grad
from tqdm import tqdm


def jacobian(f, x, batched_f = False, batched_x = False, show_progress=False):
    """Compute the Jacobian matrix/tensor Df(x).

    Batch dimensions are handled in a special way if specified.

    Parameters:
    -----------
    f (tensor): Tensor-valued function
    x (tensor): Tensor-valued variable
    batched_f (bool): The 0th axis of `f` represents different samples 
    batched_x (bool): The 0th axis of `x` represents different samples 
    show_progress (bool): Show progressbar with runtime estimate

    The generalized Jacobian for a tensor-valued function `f` of
    a tensor-valued input `x` is given by d vec(f) / d vec(x)^T, where
    the vectorization corresponds to a one-dimensional view of the

    When computing the Jacobian of a function whose output is known to
    consist of batch samples along the 0th axis with respect to batch-shaped
    parameters, it is sufficient to store the Jacobian for each
    sample/parameter pair. Speaking differently, the 0th axis of the
    computed batchwise Jacobian holds all exact Jacobians of the 0th axis
    of the function w.r.t. the zeroth axes of all parameters.   
    tensors.

    | shape(f) | shape(x) | (batched_f, batched_x) | shape(Df(x)) |
    | ------ | ------ | ------ | ------ | 
    | (f1, ...) | (x1, ...) | (False, False) | (f1 * ..., x1 * ...) 
    | (b, f1, ...) | (x1, ...) | (True, False) | (b, f1 * ..., x1 * ...) 
    | (f1, ...) | (b, x1, ...) | (False, True) | (f1 * ..., b, x1 * ...) 
    | (b, f1, ...) | (b, x1, ...) | (True, True) | (b, f1 * ..., x1 * ...) 
    """
    # full Jacobian by brute force
    jac = _jacobian_flattened_tensors(f, x, show_progress=show_progress)
    # reshape into (almost final) shape
    shape = _jacobian_shape_convention(f, x, batched_f, batched_x)
    jac = jac.view(shape)
    # slice batch axes into a single one
    if batched_f and batched_x:
        batch = list(range(shape[0]))
        jac = jac[batch,:,batch,:]
    return jac


def _jacobian_flattened_tensors(f, x, show_progress=False):
    """Compute Jacobian of flattened tensor f w.r.t. flattened tensor x. 

    Return matrix with `J[i,j] = d vec(f)[i] / d vec(x)[j]` where vec
    refers to a flattened view of a tensor-valued quantity.
    """
    jac = zeros(f.numel(), x.numel())
    progressbar = tqdm(iterable=range(f.numel()),
                       total=f.numel(),
                       desc='[exact] Jacobian',
                       disable=(not show_progress))
    f_flat = f.view(-1)
    for idx in progressbar:
        df = grad(f_flat[idx], x, create_graph=True)
        dx = None
        for d in df:
            dx = d.contiguous().view(-1) if dx is None\
                    else cat([dx, d.contiguous().view(-1)])
        jac[idx] = dx
    return jac
 

def _jacobian_shape_convention(f, x, batched_f, batched_x):
    """Return shape of the Jacobian matrix/tensor Df(x).

    Parameters:
    -----------
    f (tensor): Function
    x (tensor): Variable
    batched_f (bool): The 0th axis of `f` represents batch samples
    batched_x (bool): The 0th axis of `x` represents batch samples

    | shape(f) | shape(x) | (batched_f, batched_x) | shape(Df(x)) |
    | ------ | ------ | ------ | ------ | 
    | (f1, ...) | (x1, ...) | (False, False) | (f1 * ..., x1 * ...) 
    | (b, f1, ...) | (x1, ...) | (True, False) | (b, f1 * ..., x1 * ...) 
    | (f1, ...) | (b, x1, ...) | (False, True) | (f1 * ..., b, x1 * ...) 
    | (b, f1, ...) | (b, x1, ...) | (True, True) | (b, f1 * ..., b, x1 * ...) 

    Returns:
    --------
    jacobian_shape (tuple): Shape of the Jacobian matrix/tensor
    """
    f_total, x_total = f.numel(), x.numel()
    if (not batched_f) and (not batched_x):
        return (f_total, x_total)
    elif (batched_f) and (not batched_x):
        batch = f.size()[0]
        return (batch, f_total // batch, x_total)
    elif (not batched_f) and (batched_x):
        batch = x.size()[0]
        return (f_total, batch, x_total // batch)
    elif batched_f and batched_x:
        batch1, batch2 = f.size()[0], x.size()[0]
        if not batch1 == batch2:
            raise ValueError('f and x must have same batch dimension')
        return (batch1, f_total // batch1, batch2, x_total // batch2)
