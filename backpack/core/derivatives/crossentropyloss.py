"""Partial derivatives for cross-entropy loss."""
from math import sqrt
from typing import Dict, List, Tuple

from einops import rearrange
from torch import Tensor, diag, diag_embed, einsum, multinomial, ones_like, softmax
from torch import sqrt as torchsqrt
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot

from backpack.core.derivatives.basederivatives import BaseLossDerivatives
from backpack.utils.subsampling import subsample


class CrossEntropyLossDerivatives(BaseLossDerivatives):
    """Partial derivatives for cross-entropy loss.

    The `torch.nn.CrossEntropyLoss` operation is a composition of softmax
    and negative log-likelihood.
    """

    def _rearrange_input(
        self, probs: Tensor
    ) -> Tuple[Tensor, int, str, Dict[str, int]]:
        # if the input has additional axes: n c d1 d2 -> (n d1 d2) c
        input_dim = probs.dim()
        if input_dim >= 3:
            str_d_dims: str = str.join("", [f"d{i} " for i in range(input_dim - 2)])
            d_info: Dict[str, int] = {}
            for i in range(input_dim - 2):
                d_info[f"d{i}"] = probs.shape[2 + i]
            probs = rearrange(probs, f"n c {str_d_dims} -> (n {str_d_dims}) c")
        else:
            str_d_dims = ""
            d_info = {}
        return probs, input_dim, str_d_dims, d_info

    def _rearrange_output(
        self, sqrt_H: Tensor, input_dim, str_d_dims, d_info
    ) -> Tensor:
        if input_dim >= 3:
            sqrt_H = rearrange(
                sqrt_H, f"c1 (n {str_d_dims}) c2 -> c1 n c2 {str_d_dims}", **d_info
            )
        return sqrt_H

    def _sqrt_hessian(
        self,
        module: CrossEntropyLoss,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        subsampling: List[int] = None,
    ) -> Tensor:  # noqa: D102
        self._check_2nd_order_parameters(module)

        probs = self._get_probs(module, subsampling=subsampling)
        probs, input_dim, str_d_dims, d_info = self._rearrange_input(probs)

        tau = torchsqrt(probs)
        V_dim, C_dim = 0, 2
        Id = diag_embed(ones_like(probs), dim1=V_dim, dim2=C_dim)
        Id_tautau = Id - einsum("nv,nc->vnc", tau, tau)
        sqrt_H = einsum("nc,vnc->vnc", tau, Id_tautau)

        if module.reduction == "mean":
            N = probs.size()[0]
            sqrt_H /= sqrt(N)

        sqrt_H = self._rearrange_output(sqrt_H, input_dim, str_d_dims, d_info)
        return sqrt_H

    def _sqrt_hessian_sampled(
        self,
        module: CrossEntropyLoss,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        mc_samples: int = 1,
        subsampling: List[int] = None,
    ) -> Tensor:  # noqa: D102
        self._check_2nd_order_parameters(module)

        M = mc_samples
        C = module.input0.shape[1]

        probs = self._get_probs(module, subsampling=subsampling)
        probs, input_dim, str_d_dims, d_info = self._rearrange_input(probs)

        V_dim = 0
        probs_unsqueezed = probs.unsqueeze(V_dim).repeat(M, 1, 1)

        multi = multinomial(probs, M, replacement=True)
        classes = one_hot(multi, num_classes=C)
        classes = einsum("nvc->vnc", classes).float()

        sqrt_mc_h = (probs_unsqueezed - classes) / sqrt(M)

        if module.reduction == "mean":
            N = probs.size()[0]
            sqrt_mc_h /= sqrt(N)

        sqrt_mc_h = self._rearrange_output(sqrt_mc_h, input_dim, str_d_dims, d_info)
        return sqrt_mc_h

    def _sum_hessian(self, module, g_inp, g_out):
        self._check_2nd_order_parameters(module)

        probs = self._get_probs(module)
        probs, input_dim, str_d_dims, d_info = self._rearrange_input(probs)
        sum_H = diag(probs.sum(0)) - einsum("bi,bj->ij", (probs, probs))

        if module.reduction == "mean":
            N = probs.size()[0]
            sum_H /= N

        return sum_H

    # TODO: discuss whether rearrange should also be applied in this function
    # TODO: discuss why this has no test in test_derivatives.py
    def _make_hessian_mat_prod(self, module, g_inp, g_out):
        """Multiplication of the input Hessian with a matrix."""
        self._check_2nd_order_parameters(module)

        probs = self._get_probs(module)

        def hessian_mat_prod(mat):
            Hmat = einsum("bi,cbi->cbi", (probs, mat)) - einsum(
                "bi,bj,cbj->cbi", (probs, probs, mat)
            )

            if module.reduction == "mean":
                N = module.input0.shape[0]
                Hmat /= N

            return Hmat

        return hessian_mat_prod

    def hessian_is_psd(self):
        """Return whether cross-entropy loss Hessian is positive semi-definite."""
        return True

    def _get_probs(
        self, module: CrossEntropyLoss, subsampling: List[int] = None
    ) -> Tensor:
        """Compute the softmax probabilities from the module input.

        Args:
            module: cross-entropy loss with I/O.
            subsampling: Indices of samples to be considered. Default of ``None`` uses
                the full mini-batch.

        Returns:
            Softmax probabilites
        """
        # TODO question: does this imply the assumption that it is always
        # loss(model(x), y) and not loss(y, model(x)) ???
        input0 = subsample(module.input0, subsampling=subsampling)
        return softmax(input0, dim=1)

    def _check_2nd_order_parameters(self, module):
        """Verify that the parameters are supported by 2nd-order quantities.

        Attributes:
            module (torch.nn.CrossEntropyLoss): Extended CrossEntropyLoss module

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
