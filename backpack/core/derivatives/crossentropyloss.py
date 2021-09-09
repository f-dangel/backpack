"""Partial derivatives for cross-entropy loss."""
from itertools import product as itertools_product
from math import sqrt
from typing import Callable, Dict, Iterable, List, Tuple, Union

from einops import rearrange
from torch import Tensor, diag, diag_embed, einsum, eye, multinomial, ones_like, softmax
from torch import sqrt as torchsqrt
from torch import zeros
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot

from backpack.core.derivatives.basederivatives import BaseLossDerivatives
from backpack.utils.subsampling import subsample


class CrossEntropyLossDerivatives(BaseLossDerivatives):
    """Partial derivatives for cross-entropy loss.

    The `torch.nn.CrossEntropyLoss` operation is a composition of softmax
    and negative log-likelihood.
    """

    def _sqrt_hessian(
        self,
        module: CrossEntropyLoss,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        subsampling: List[int] = None,
    ) -> Tensor:
        self._check_2nd_order_parameters(module)

        probs = self._get_probs(module, subsampling=subsampling)
        probs, *rearrange_info = self._merge_batch_and_additional(probs)

        tau = torchsqrt(probs)
        V_dim, C_dim = 0, 2
        Id = diag_embed(ones_like(probs), dim1=V_dim, dim2=C_dim)
        Id_tautau = Id - einsum("nv,nc->vnc", tau, tau)
        sqrt_H = einsum("nc,vnc->vnc", tau, Id_tautau)

        if module.reduction == "mean":
            sqrt_H /= sqrt(self._get_mean_normalization(module.input0))

        sqrt_H = self._ungroup_batch_and_additional(sqrt_H, *rearrange_info)
        sqrt_H = self._expand_sqrt_h(sqrt_H, module, subsampling)
        return sqrt_H

    def _sqrt_hessian_sampled(
        self,
        module: CrossEntropyLoss,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mc_samples: int = 1,
        subsampling: List[int] = None,
    ) -> Tensor:
        self._check_2nd_order_parameters(module)

        M = mc_samples
        C = module.input0.shape[1]

        probs = self._get_probs(module, subsampling=subsampling)
        probs, *rearrange_info = self._merge_batch_and_additional(probs)

        V_dim = 0
        probs_unsqueezed = probs.unsqueeze(V_dim).repeat(M, 1, 1)

        multi = multinomial(probs, M, replacement=True)
        classes = one_hot(multi, num_classes=C)
        classes = einsum("nvc->vnc", classes).float()

        sqrt_mc_h = (probs_unsqueezed - classes) / sqrt(M)

        if module.reduction == "mean":
            sqrt_mc_h /= sqrt(self._get_mean_normalization(module.input0))

        sqrt_mc_h = self._ungroup_batch_and_additional(sqrt_mc_h, *rearrange_info)
        return sqrt_mc_h

    def _sum_hessian(
        self, module: CrossEntropyLoss, g_inp: Tuple[Tensor], g_out: Tuple[Tensor]
    ) -> Tensor:
        self._check_2nd_order_parameters(module)

        probs = self._get_probs(module)

        if probs.dim() == 2:
            diagonal = diag(probs.sum(0))
            sum_H = diagonal - einsum("nc,nd->cd", probs, probs)
        else:
            diagonal = diag(probs.sum(0).flatten()).reshape(
                *probs.shape[1:], *probs.shape[1:]
            )
            sum_H = (
                diagonal
                - einsum(
                    "ncx,ndy,xy->cxdy",
                    probs.flatten(2),
                    probs.flatten(2),
                    eye(
                        probs.shape[2:].numel(),
                        probs.shape[2:].numel(),
                        device=probs.device,
                    ),
                ).reshape(*probs.shape[1:], *probs.shape[1:])
            )

        if module.reduction == "mean":
            sum_H /= self._get_mean_normalization(module.input0)

        return sum_H

    def _make_hessian_mat_prod(
        self, module: CrossEntropyLoss, g_inp: Tuple[Tensor], g_out: Tuple[Tensor]
    ) -> Callable[[Tensor], Tensor]:
        self._check_2nd_order_parameters(module)

        probs = self._get_probs(module)

        def hessian_mat_prod(mat):
            Hmat = einsum("nc...,vnc...->vnc...", probs, mat) - einsum(
                "nc...,nd...,vnd...->vnc...", probs, probs, mat
            )

            if module.reduction == "mean":
                Hmat /= self._get_mean_normalization(module.input0)

            return Hmat

        return hessian_mat_prod

    def hessian_is_psd(self) -> bool:
        """Return whether cross-entropy loss Hessian is positive semi-definite.

        Returns:
            True
        """
        return True

    @staticmethod
    def _get_probs(module: CrossEntropyLoss, subsampling: List[int] = None) -> Tensor:
        """Compute the softmax probabilities from the module input.

        Args:
            module: cross-entropy loss with I/O.
            subsampling: Indices of samples to be considered. Default of ``None`` uses
                the full mini-batch.

        Returns:
            Softmax probabilites
        """
        input0 = subsample(module.input0, subsampling=subsampling)
        return softmax(input0, dim=1)

    def _check_2nd_order_parameters(self, module: CrossEntropyLoss) -> None:
        """Verify that the parameters are supported by 2nd-order quantities.

        Args:
            module: Extended CrossEntropyLoss module

        Raises:
            NotImplementedError: If module's setting is not implemented.
        """
        implemented_ignore_index = -100
        implemented_weight = None

        if module.ignore_index != implemented_ignore_index:
            raise NotImplementedError(
                "Only default ignore_index ({}) is implemented, got {}".format(
                    implemented_ignore_index, module.ignore_index
                )
            )

        if module.weight != implemented_weight:
            raise NotImplementedError(
                "Only default weight ({}) is implemented, got {}".format(
                    implemented_weight, module.weight
                )
            )

    @staticmethod
    def _merge_batch_and_additional(
        probs: Tensor,
    ) -> Tuple[Tensor, str, Dict[str, int]]:
        """Rearranges the input if it has additional axes.

        Treat additional axes like batch axis, i.e. group ``n c d1 d2 -> (n d1 d2) c``.

        Args:
            probs: the tensor to rearrange

        Returns:
            a tuple containing
                - probs: the rearranged tensor
                - str_d_dims: a string representation of the additional dimensions
                - d_info: a dictionary encoding the size of the additional dimensions
        """
        leading = 2
        additional = probs.dim() - leading

        str_d_dims: str = "".join(f"d{i} " for i in range(additional))
        d_info: Dict[str, int] = {
            f"d{i}": probs.shape[leading + i] for i in range(additional)
        }

        probs = rearrange(probs, f"n c {str_d_dims} -> (n {str_d_dims}) c")

        return probs, str_d_dims, d_info

    @staticmethod
    def _ungroup_batch_and_additional(
        tensor: Tensor, str_d_dims, d_info, free_axis: int = 1
    ) -> Tensor:
        """Rearranges output if it has additional axes.

        Used with group_batch_and_additional.

        Undoes treating additional axes like batch axis and assumes an number of
        additional free axes (``v``) were added, i.e. un-groups
        ``v (n d1 d2) c -> v n c d1 d2``.

        Args:
            tensor: the tensor to rearrange
            str_d_dims: a string representation of the additional dimensions
            d_info: a dictionary encoding the size of the additional dimensions
            free_axis: Number of free leading axes. Default: ``1``.

        Returns:
            the rearranged tensor

        Raises:
            NotImplementedError: If ``free_axis != 1``.
        """
        if free_axis != 1:
            raise NotImplementedError(f"Only supports free_axis=1. Got {free_axis}.")

        return rearrange(
            tensor, f"v (n {str_d_dims}) c -> v n c {str_d_dims}", **d_info
        )

    @staticmethod
    def _expand_sqrt_h(
        sqrt_h: Tensor, module: CrossEntropyLoss, subsampling: Union[None, List[int]]
    ) -> Tensor:
        """Expands the square root hessian if CrossEntropyLoss has additional axes.

        In the case of e.g. two additional axes, the input is [N,C,A,B].
        In CrossEntropyLoss the additional axes are treated independently.
        Therefore the intermediate result has shape [C,N,C,A,B].
        In subsequent calculations the additional axes are not independent anymore.
        The required shape for sqrt_h_full is then [C*A*B,N,C,A,B].
        Due to the independence, sqrt_h lives on the diagonal of sqrt_h_full.

        Args:
            sqrt_h: intermediate result, shape [C,N,C,A,B]
            module: module, to extract input shape
            subsampling: subsampling to determine length of batch axis

        Returns:
            sqrt_h_full, shape [C*A*B,N,C,A,B], sqrt_h on diagonal.
        """
        n_standard_axes = 2
        n_additional_axes = module.input0.dim() - n_standard_axes

        if n_additional_axes > 0:
            shape_input = module.input0.shape
            N_samples = shape_input[0] if subsampling is None else len(subsampling)
            sqrt_H_full = zeros(*shape_input[1:], N_samples, *shape_input[1:])
            ranges_additional_axes: List[Iterable, ...] = [
                range(shape_input[n_standard_axes + i])
                for i in range(n_additional_axes)
            ]
            index_additional_axes: Tuple[int]
            for index_additional_axes in itertools_product(*ranges_additional_axes):
                selection_sqrt_H = (slice(None),) * 3 + index_additional_axes
                selection_target = (
                    (slice(None),)
                    + index_additional_axes
                    + (slice(None),) * 2
                    + index_additional_axes
                )
                sqrt_H_full[selection_target] = sqrt_h[selection_sqrt_H]
            return sqrt_H_full.flatten(0, n_additional_axes)
        return sqrt_h

    @staticmethod
    def _get_mean_normalization(input: Tensor) -> int:
        """Get normalization constant used with reduction='mean'.

        Args:
            input: Input to the cross-entropy module.

        Returns:
            Divisor for mean reduction.
        """
        return input.numel() // input.shape[1]
