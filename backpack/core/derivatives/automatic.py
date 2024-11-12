"""Automatic derivative implementation via ``torch.autograd``."""

from abc import abstractmethod
from typing import Dict, List, Optional, Protocol, Tuple, Union

from torch import Tensor, allclose, cat, enable_grad, stack
from torch.autograd import grad
from torch.nn import Module, Parameter

from backpack.core.derivatives import shape_check
from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.utils.subsampling import subsample


class ForwardCallable(Protocol):
    """Type-annotation for functions performing a forward pass."""

    def __call__(
        self,
        x: Tensor,
        *params_args: Union[Parameter, Tensor],
        **params_kwargs: Union[Parameter, Tensor, None],
    ) -> Tensor: ...  # noqa: D102


class AutomaticDerivatives(BaseParameterDerivatives):
    """Implements derivatives for an arbitrary layer using ``torch.autograd``.

    This class can be used to support new layers without implementing their
    derivatives. However, this comes at the cost of performance, since the
    autograd-based implementation is often not as efficient as a hand-crafted one.

    Attributes:
        BATCH_AXIS: Index of the layer input's batch axis. Default: ``0``.
    """

    BATCH_AXIS: int = 0

    @staticmethod
    @abstractmethod
    def as_functional(module: Module) -> ForwardCallable:
        """Return a function that performs the layer's forward pass.

        Args:
            module: Layer for which to return the forward function.

        Returns:
            Function that performs the forward pass of the layer and returns a tensor
            representing the result. First argument must be the input tensor, and
            subsequent keyword arguments must be the layer's parameters.

        Note:
            One way to automate this procedure would be via
            ``torch.func.functional_call``. However, this does not work at the moment
            because the passed layer has hooks. For now, this function must thus
            be specified explicitly.
        """
        raise NotImplementedError("Must be implemented by a child class.")

    @classmethod
    def forward_pass(
        cls, module: Module, subsampling: Optional[List[int]] = None
    ) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
        """Perform a forward pass through the layer.

        Args:
            module: Layer for which to perform the forward pass.
            subsampling: Indices of the batch axis to keep. If ``None``, all indices
                are kept.Default: ``None``.

        Returns:
            The sub-sampled tensor used as input to the forward pass, the parameters,
            and the output.

        Raises:
            RuntimeError: If the forward function produces a different output than the
                layer's forward pass.
        """
        # Create an independent copy of the layer's input and parameters
        input0 = module.input0.clone().detach()
        input0 = subsample(input0, dim=cls.BATCH_AXIS, subsampling=subsampling)
        params = {
            name: param.clone().detach() for name, param in module.named_parameters()
        }

        # turn on autograd for input and parameters
        input0.requires_grad_(True)
        for param in params.values():
            param.requires_grad_(True)

        forward_fn = cls.as_functional(module)
        output = forward_fn(input0, **params)

        # make sure the layer's re-created output matches the output from the
        # initial forward pass
        if not allclose(
            output,
            subsample(module.output, dim=cls.BATCH_AXIS, subsampling=subsampling),
        ):
            raise RuntimeError(
                "Forward function used inside `AutogradDerivatives` produced a "
                + "different output than the module's forward pass. This indicates "
                + "1) the layer is non-deterministic and cannot be supported by "
                + "`AutogradDerivatives`, or 2) `.as_functional` is incorrect."
            )

        return input0, params, output

    # NOTE Explicitly turn on autodiff as this function is called during a
    # backward pass.
    @enable_grad()
    def _jac_t_mat_prod(
        self,
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mat: Tensor,
        subsampling: Optional[List[int]] = None,
    ) -> Tensor:
        # regenerate computation graph for differentiation
        input0, _, output = self.forward_pass(module, subsampling=subsampling)
        return grad(output, input0, grad_outputs=mat, is_grads_batched=True)[0]

    # NOTE Explicitly turn on autodiff as this function is called during a
    # backward pass.
    @enable_grad()
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
        batch_size = module.input0.shape[self.BATCH_AXIS]
        subsampling = list(range(batch_size)) if subsampling is None else subsampling

        # contains the MJPs for each sample along the batch dimension
        sample_vjps = []

        for sample_idx, sample in enumerate(subsampling):
            # regenerate computation graph for differentiation
            _, params, output = self.forward_pass(module, subsampling=[sample])
            # shape [V, *module.param_str.shape]
            vjps = grad(
                output,
                params[param_str],
                grad_outputs=mat[:, [sample_idx]],
                is_grads_batched=True,
            )[0]
            sample_vjps.append(vjps.sum(self.BATCH_AXIS) if sum_batch else vjps)

        # shape [V, B, *module.param_str.shape] or [V, *module.param_str.shape]
        return (
            cat(sample_vjps, dim=self.BATCH_AXIS)
            if sum_batch
            else stack(sample_vjps, dim=self.BATCH_AXIS + 1)
        )
