"""Support parameter splitting for a sequence of parallel modules."""

from warnings import warn
from ..sequential import HBPSequential
from .parallel import HBPParallel
from ..linear import HBPLinear
from .linear import HBPParallelLinear
from ..relu import HBPReLU
from ..sigmoid import HBPSigmoid
from ..combined import HBPCompositionActivationLinear
from .combined import HBPParallelCompositionActivationLinear


class HBPParallelSequential(HBPSequential):
    """Handle splitting for a sequence of parallel modules."""
    def __init__(self, num_blocks, *layers):
        """Convert to parallel before if known."""
        converted = [self._convert_to_parallel(layer,
                                               num_blocks)
                     for layer in layers]
        super().__init__(*converted)

    def _convert_to_parallel(self, layer, num_blocks):
        """Convert a layer to a parallelized form.
        
        Layers that will not be converted do not end up with the
        specified split into `num_blocks`.
        """
        target_cls = self._find_target_class(layer.__class__)

        if target_cls is None:
            warn('Conversion to parallel: The following layer will be'
                 'left unchanged:r\n{}'.format(layer))
            return layer
        else:
            return target_cls(layer, num_blocks)

    # known conversions for layers to parallel equivalents
    conversions = {
                   HBPLinear:
                    HBPParallelLinear,
                   HBPCompositionActivationLinear:
                    HBPParallelCompositionActivationLinear,
                  }
    # no conversions will happen for these layers
    no_conversion = [
                     HBPReLU,
                     HBPSigmoid,
                     HBPParallel,
                     HBPSequential,
                    ]

    @classmethod
    def _find_target_class(cls, layer_cls):
        """Find the target class to convert to.

        Returns:
        --------
        (class)
            Target class to convert to, `None` if layer does not
            have to be converted

        Raises:
        -------
        (ValueError)
            If no conversion rule for `layer_cls` is found,
            or multiple conversions are possible
        """
        # check if in no conversion
        if layer_cls in cls.no_conversion:
            return None
        subcls = [cls.conversions[sub] for sub in cls.conversions.keys()
                  if issubclass(layer_cls, sub)]
        if len(subcls) == 1:
            return subcls[0]
        elif len(subcls) > 1:
            raise ValueError('Found multiple target classes to convert'
                             '{} to: {}\nSupported conversions:\n{}'
                             .format(layer_cls, subcls,
                                     cls._supported_conversions()))
        else:
            raise ValueError('No conversion strategy found. Supported'
                             ' conversions:\n{}'
                             .format(cls._supported_conversions()))
 
    @classmethod
    def _supported_conversions(cls):
        """Return a string listing the conversions."""
        result = 'CONVERSIONS:\n'
        result.append('-' * len(result))
        for key, val in cls.conversions.items():
            result.append('\n{:25s} -> {:25s}'.format(key, val))
        for val in cls.no_conversions:
            result.append('\n{:25s} -> {:25s}'.format(val, val))
        return result
