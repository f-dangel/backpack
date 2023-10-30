"""Automatic derivative implementation via ``torch.autograd``."""

from abc import abstractmethod
from typing import Dict, List, Optional, Protocol, Tuple, Union

from torch import Tensor, allclose, enable_grad, stack
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
        **params_kwargs: Union[Parameter, Tensor, None]
    ) -> Tensor:
        ...


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

        # ``mat`` consists of ``V`` vectors of shape ``[*module.output.shape]``
        vjps = [
            grad(output, input0, v, retain_graph=idx != mat.shape[0] - 1)[0]
            for idx, v in enumerate(mat)
        ]

        return stack(vjps)  # shape [V, *module.input0.shape]

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
        batch_size = module.input0.shape[self.BATCH_AXIS]
        subsampling = list(range(batch_size)) if subsampling is None else subsampling

        # contains the MJPs for each sample along the batch dimension
        sample_vjps = []

        # ``mat`` consists of ``V`` vectors of shape ``[*module.output.shape]``
        num_vecs = mat.shape[0]

        for sample_idx, sample in enumerate(subsampling):
            # regenerate computation graph for differentiation
            _, params, output = self.forward_pass(module, subsampling=[sample])

            vjps = [
                grad(
                    output,
                    params[param_str],
                    v,
                    retain_graph=v_idx != num_vecs - 1,
                )[0]
                for v_idx, v in enumerate(mat[:, [sample_idx]])
            ]

            sample_vjps.append(stack(vjps))  # shape [V, *module.param_str.shape]

        sample_vjps = stack(sample_vjps, dim=1)

        if sum_batch:
            sample_vjps = sample_vjps.sum(1)

        return sample_vjps  # shape [V, B, *module.param_str.shape] or [V, B, *module.param_str.shape]
