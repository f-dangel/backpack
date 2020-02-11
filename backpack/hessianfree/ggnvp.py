from .hvp import hessian_vector_product
from .lop import L_op
from .rop import R_op


def ggn_vector_product(loss, output, model, v):
    """
    Multiplies the vector `v` with the Generalized Gauss-Newton,
    `ggn_v = J.T @ H @ J @ v`

    where `J` is the Jacobian of `output` w.r.t. `model.parameters()`
    and `H` is the Hessian of `loss` w.r.t. `output`.

    Example usage:
    ```
    X, Y = data()
    model = torch.nn.Linear(784, 10)
    lossfunc = torch.nn.CrossEntropyLoss()

    output = model(X)
    loss = lossfunc(output, Y)

    v = list([torch.randn_like(p) for p in model.parameters])

    GGNv = ggn_vector_product(loss, output, model, v)
    ```

    Parameters:
    -----------
        loss: torch.Tensor
        output: torch.Tensor
        model: torch.nn.Module
        v: [torch.Tensor]
            List of tensors matching the sizes of model.parameters()
    """
    return ggn_vector_product_from_plist(loss, output, list(model.parameters()), v)


def ggn_vector_product_from_plist(loss, output, plist, v):
    Jv = R_op(output, plist, v)
    HJv = hessian_vector_product(loss, output, Jv)
    JTHJv = L_op(output, plist, HJv)
    return JTHJv
