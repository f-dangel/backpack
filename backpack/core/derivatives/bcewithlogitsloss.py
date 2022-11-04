"""NLL extention for BCEWithLogits Loss."""

from math import sqrt
from typing import List, Tuple

from torch import Size, Tensor, sigmoid
from torch.distributions import Binomial
from torch.nn import BCEWithLogitsLoss

from backpack.core.derivatives.nll_base import NLLLossDerivatives
from backpack.utils.subsampling import subsample


class BCELossWithLogitsDerivatives(NLLLossDerivatives):
    """Derivatives of the BCEWithLogits Loss."""

    def __init__(self, use_autograd: bool = False):
        """Initialization for BCEWithLogitsLoss derivative.

        Args:
            use_autograd: Compute gradients with autograd (rather than manual)
                Defaults to ``False`` (manual computation).
        """
        super().__init__(use_autograd=use_autograd)

    def _verify_support(self, module: BCEWithLogitsLoss):
        """Verification of module support for BCEWithLogitsLoss.

        Currently BCEWithLogitsLoss only supports binary target tensors,
        2D inputs, and default parameters.

        Args:
            module: BCEWithLogitsLoss module
        """
        self._check_binary(module)
        self._check_is_default(module)
        self._check_input_dims(module)

    def _check_binary(self, module: BCEWithLogitsLoss):
        """Raises exception if labels are not binary.

        Args:
            module: BCEWithLogitsLoss module

        Raises:
            NotImplementedError: if labels are non-binary.
        """
        if any(x not in [0, 1] for x in module.input1.flatten()):
            raise NotImplementedError(
                "Only binary targets (0 and 1) are currently supported."
            )

    def _check_is_default(self, module: BCEWithLogitsLoss):
        """Raises exception if module parameters are not default.

        Args:
            module: BCEWithLogitsLoss module

        Raises:
            NotImplementedError: if module parameters non-default.
        """
        if module.weight is not None:
            raise NotImplementedError("Only None weight is currently supported.")
        if module.pos_weight is not None:
            raise NotImplementedError("Only None pos_weight is currently supported.")

    def _check_input_dims(self, module: BCEWithLogitsLoss):
        """Raises an exception if the shapes of the input are not supported.

        Args:
            module: BCEWithLogitsLoss module

        Raises:
            NotImplementedError: if input is not a batch of scalars.
        """
        if module.input0.dim() != 2:
            raise NotImplementedError("Only 2D inputs are currently supported.")
        if module.input0.shape[1] != 1:
            raise NotImplementedError(
                "Only scalar-valued predictions are currently supported."
            )

    def _make_distribution(self, subsampled_input: Tensor) -> Binomial:
        """Make the sampling distribution for the NLL loss form of BCEWithLogits.

        The BCEWithLogitsLoss ∝ ∑ᵢ₌₁ⁿ Yᵢ log 𝜎(xᵢ) + (1 − Yᵢ) log(1− 𝜎(xᵢ)).
        The log likelihood of the Binomial distribution is
        Yᵢ log p(xᵢ) + (1 − Yᵢ) log(1 − p(xᵢ)), so these are equivalent if
        p(xᵢ) = 𝜎(xᵢ).

        Args:
            subsampled_input: input after subsampling

        Returns:
            Binomial distribution with sigmoid probabilities from the subsampled_input.
        """
        return Binomial(probs=subsampled_input.sigmoid())

    def _compute_sampled_grads_manual(
        self, subsampled_input: Tensor, mc_samples: int
    ) -> Tensor:
        """Manually compute gradients from sampled targets.

        Let fₙ ∈ ℝ and yₙ ∈ {0, 1} ∼ p(y | fₙ) and σ(fₙ) the softmax probability.
        Then the gradient is ∇ℓ(fₙ, yₙ) = σ(fₙ) - fₙ.

        Args:
            subsampled_input: input after subsampling
            mc_samples: number of samples

        Returns:
            Gradient samples
        """
        probs = subsampled_input.sigmoid()
        expand_dims = [mc_samples] + probs.dim() * [-1]
        probs_unsqeezed = probs.unsqueeze(0).expand(*expand_dims)  # [V N 1]

        distribution = self._make_distribution(subsampled_input)
        samples = distribution.sample(Size([mc_samples]))  # [V N 1]

        return probs_unsqeezed - samples

    @staticmethod
    def _get_mean_normalization(input: Tensor) -> int:
        return input.shape[0]

    def _sqrt_hessian(
        self,
        module: BCEWithLogitsLoss,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        subsampling: List[int],
    ) -> Tensor:  # noqa: D102
        """Return a symmetric factorization of the loss Hessian.

        # noqa: DAR101

        Let fₙ ∈ ℝ be the input and yₙ ∈ [0; 1] be the label, and σ(fₙ) ∈ (0;
        1) be the sigmoid probability. Then, the gradient ∇ℓ(fₙ, yₙ) w.r.t. fₙ
        is ∇ℓ(fₙ, yₙ) = σ(fₙ) - yₙ, and the Hessian ∇²ℓ(fₙ, yₙ) w.r.t. fₙ is
        ∇²ℓ(fₙ, yₙ) = σ'(fₙ) = σ(fₙ) (1 - σ(fₙ)). Consequently, the (scalar)
        Hessian square root is √(σ(fₙ) (1 - σ(fₙ))).

        Returns:
            Hessian square root factorization of shape ``[1, N, 1]`` where ``N``
            corresponds to the (subsampled) batch size.
        """
        self._check_is_default(module)
        self._check_input_dims(module)

        input0 = subsample(module.input0, subsampling=subsampling)
        sigma = sigmoid(input0).unsqueeze(0)

        sqrt_H = (sigma * (1 - sigma)).sqrt()

        if module.reduction == "mean":
            sqrt_H /= sqrt(self._get_mean_normalization(module.input0))

        return sqrt_H

    def hessian_is_psd(self) -> bool:
        """Return whether the Hessian is PSD.

        Let fₙ ∈ ℝ be the input and yₙ ∈ [0; 1] be the label, and σ(fₙ) ∈ (0;
        1) be the sigmoid probability. The Hessian ∇²ℓ(fₙ, yₙ) w.r.t. fₙ is
        ∇²ℓ(fₙ, yₙ) = σ'(fₙ) = σ(fₙ) (1 - σ(fₙ)) > 0. Hence, the Hessian is PSD.

        Returns:
            True
        """
        return True
