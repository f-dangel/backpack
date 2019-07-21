import torch

from .rop import R_op


def hessian_vector_product(f, params, v, detach=True):
    """
    Multiplies the vector `v` with the Hessian,
    `v = H @ v`

    where `H` is the Hessian of `f` w.r.t. `params`.

    Example usage:
    ```
    X, Y = data()
    model = torch.nn.Linear(784, 10)
    lossfunc = torch.nn.CrossEntropyLoss()

    loss = lossfunc(output, Y)

    v = list([torch.randn_like(p) for p in model.parameters])

    Hv = hessian_vector_product(loss, list(model.parameters()), v)
    ```

    Parameters:
    -----------
        f: torch.Tensor
        params: torch.Tensor or [torch.Tensor]
        v: torch.Tensor or [torch.Tensor]
            Shapes must match `params`
        detach: Bool, optional
            Whether to detach the output from the computation graph
            (default: True)
    """
    df_dx = torch.autograd.grad(f, params, create_graph=True, retain_graph=True)
    Hv = R_op(df_dx, params, v)

    if detach:
        return tuple([j.detach() for j in Hv])
    else:
        return Hv
