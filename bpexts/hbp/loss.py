"""Compute batch-summed Hessian of the loss w.r.t network outputs."""


from torch import cat
from torch.autograd import grad


def batch_summed_hessian(f, x):
    """Compute summed Hessian of `f` with respect to `x`.

    The 0th dimension of `x` is considered the batch dimension over
    which the sum is taken.

    Parameters:
    -----------
    f (scalar): Function
    x (tensor): Parameters
    """
    if not f.numel() == 1:
        raise ValueError('Expect scalar function')
    dim = x.numel() // x.size()[0]
    df, = grad(f, x, create_graph=True)
    # summed over batch samples
    df = df.sum(0)
    # vector for collecting all second derivatives
    d2f = None
    for d in df:
        # take derivative with respect to all inputs
        d2, = grad(d, x, create_graph=True)
        # average over batch sample
        d2 = d2.sum(0)
        # concatenate to result
        d2f = d2.contiguous().view(-1) if d2f is None\
            else cat([d2f, d2.contiguous().view(-1)])
    return d2f.reshape(dim, dim)
