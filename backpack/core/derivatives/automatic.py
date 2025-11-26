"""Automatic derivative implementation via ``torch.autograd``."""

from copy import deepcopy
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

from torch import Tensor
from torch.func import functional_call, vjp, vmap
from torch.nn import Module

from backpack.core.derivatives import shape_check
from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.utils.subsampling import subsample


class AutomaticDerivatives(BaseParameterDerivatives):
    """Implements derivatives for an arbitrary layer using ``torch.func``.

    This class can be used to support new layers without implementing their
    derivatives by hand. However, this comes at the cost of performance, since the
    autograd-based implementation is often not as efficient as a hand-crafted one
    and re-evaluates the forward pass through the layer.

    Attributes:
        BATCH_AXIS: Index of the layer input's batch axis. Default: ``0``.
    """

    BATCH_AXIS: int = 0

    @staticmethod
    def clone_without_hooks(module: Module) -> Module:
        """Create a copy of module without BackPACK hooks.

        Args:
            module: Module to be cloned.

        Returns:
            Cloned module without BackPACK hooks.
        """
        # Temporarily remove input and output to avoid error when calling deepcopy
        input0, output = module.input0, module.output
        delattr(module, "input0")
        delattr(module, "output")

        # Clone the module and remove BackPACK's hooks
        clean = deepcopy(module)
        clean._forward_hooks.clear()
        clean._backward_hooks.clear()

        # Restore input and output in the original module
        module.input0, module.output = input0, output

        return clean

    def as_functional(
        self, module: Module, param_name: Optional[str] = None
    ) -> Union[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor], Tensor]]:
        """Return a function that performs the layer's forward pass on a single datum.

        Args:
            module: Layer for which to return the forward function.
            param_name: If specified, the name of a parameter of the module that
                will be passed as second argument to the returned function.
                Default: ``None`` (all parameters are frozen in the module).

        Returns:
            Function that performs the forward pass of the layer and returns a tensor
            representing the result. First argument is the un-batched input tensor.
            If `param_name` is specified, the second argument corresponds to the
            parameter.
        """
        module = self.clone_without_hooks(module)
        parameters = dict(module.named_parameters())
        buffers = dict(module.named_buffers())

        if param_name is None:

            def f(x: Tensor) -> Tensor:
                return functional_call(module, {**parameters, **buffers}, x)

        else:
            parameters.pop(param_name)

            def f(x: Tensor, param: Tensor) -> Tensor:
                return functional_call(
                    module, {**parameters, **buffers, param_name: param}, x
                )

        return f

    def _jac_t_mat_prod(
        self,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        subsampling: Optional[List[int]] = None,
    ) -> Tensor:
        f = vmap(self.as_functional(module))  # {x_n} -> {f(x_n)}

        X = subsample(module.input0, dim=self.BATCH_AXIS, subsampling=subsampling)
        _, vjp_func = vjp(f, X)  # {v_n} -> {Jf(x_n)^T v_n}
        # vmap over matrix columns
        vmp_func = vmap(vjp_func)
        (vmp,) = vmp_func(mat)

        return vmp

    @shape_check.param_mjp_accept_vectors
    def param_mjp(
        self,
        param_str: str,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        sum_batch: bool = True,
        subsampling: Optional[List[int]] = None,
    ) -> Tensor:
        """Compute matrix-Jacobian products (MJPs) of the module w.r.t. a parameter.

        Handles both vector and matrix inputs. Preserves input format in output.

        Args:
            param_str: Attribute name under which the parameter is stored in the module.
            module: Module whose Jacobian will be applied. Must provide access to IO.
            g_inp: Gradients w.r.t. module input.
            g_out: Gradients w.r.t. module output.
            mat: Matrix the Jacobian will be applied to. Has shape
                ``[V, *module.output.shape]`` (matrix case) or same shape as
                ``module.output`` (vector case). If used with subsampling, has dimension
                len(subsampling) instead of batch size along the batch axis.
            sum_batch: Sum out the MJP's batch axis. Default: ``True``.
            subsampling: Indices of samples along the output's batch dimension that
                should be considered. Defaults to ``None`` (use all samples).

        Returns:
            Matrix-Jacobian products. Has shape ``[V, *param_shape]`` when batch
            summation is enabled (same shape as parameter in the vector case). Without
            batch summation, the result has shape ``[V, N, *param_shape]`` (vector case
            has shape ``[N, *param_shape]``). If used with subsampling, the batch size N
            is replaced by len(subsampling).
        """
        f = self.as_functional(
            module, param_name=param_str
        )  # (x, param) -> f(x, param)

        def param_vjp(x, param, v) -> Tensor:
            f_x = partial(f, x)  # param -> f(x, param)
            _, vjp_func = vjp(f_x, param)  # v -> Jf(x, param)^T v
            (mjp,) = vjp_func(v)
            return mjp

        # vectorize over data points: ({x_n}, param, {v_n}) -> {Jf(x_n, param)^T v_n}
        param_vjp = vmap(param_vjp, in_dims=(self.BATCH_AXIS, None, self.BATCH_AXIS))
        # vectorize over matrix columns
        param_mjp = vmap(param_vjp, in_dims=(None, None, 0))

        X = subsample(module.input0, dim=self.BATCH_AXIS, subsampling=subsampling)
        mjp = param_mjp(X, getattr(module, param_str), mat)

        if sum_batch:
            mjp = mjp.sum(dim=self.BATCH_AXIS + 1)

        return mjp
