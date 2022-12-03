"""BackPACK extension for Monte-Carlo estimate of the Hessian diagonal.

Based on

- Martens, J., Sutskever, I., & Swersky, K. (2012). Estimating the hessian by
  back-propagating curvature. (ICML).
"""

from typing import Tuple, Union

from torch import Size, Tensor, bernoulli, device, normal, ones
from torch.nn import Linear, MSELoss, Sigmoid

from backpack.extensions.secondorder.base import SecondOrderBackpropExtension
from backpack.extensions.secondorder.diag_hessian_mc import activations, linear, losses


class DiagHessianMC(SecondOrderBackpropExtension):
    """BackPACK extension that approximates the Hessian diagonal.

    For a more precise but slower alternative,
    see :py:meth:`backpack.extensions.DiagHessian`.
    """

    def __init__(self, mc_samples: int = 1, mc: str = "Normal"):
        """Store savefield and mappings between layers and module extensions.

        Args:
            mc_samples: Number of samples. Default: ``1``.
            mc: Type of random vectors. Must be ``'Normal'`` or ``'Binomial'``.
                Default: ``'Gaussian'``.
        """
        self._mc_samples = mc_samples
        self._mc = mc

        super().__init__(
            savefield="diag_h_mc",
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses.DiagHMCMSELoss(),
                Linear: linear.DiagHMCLinear(),
                Sigmoid: activations.DiagHMCSigmoid(),
            },
        )

    def get_num_mc_samples(self) -> int:
        """Returns number of Monte-Carlo samples.

        Returns:
            number of Monte-Carlo samples
        """
        return self._mc_samples

    def get_random_vector(self, size: Union[Tuple[int], Size], dev: device) -> Tensor:
        """Draw random vector."""
        shape = (self._mc_samples,) + tuple(size)
        if self._mc == "Normal":
            mean, std = 0.0, 1.0
            return normal(mean, std, shape, device=dev)
        elif self._mc == "Bernoulli":
            probs = 0.5 * ones(shape)
            return 2 * (bernoulli(probs).to(dev) - 0.5)

        raise ValueError(f"Invalid MC argument {self._mc}")
