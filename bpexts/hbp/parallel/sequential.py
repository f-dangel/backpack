"""Support parameter splitting for a sequence of parallel modules."""

from warnings import warn
from ..sequential import HBPSequential
from .parallel import HBPParallel
from ..linear import HBPLinear
from .linear import HBPParallelLinear
from ..combined import HBPCompositionActivationLinear
from .combined import HBPParallelCompositionActivationLinear


class HBPParallelSequential(HBPSequential):
    """Handle splitting for a sequence of parallel modules."""
    def __init__(self, *layers):
        """Convert to parallel before if known."""
        converted = [convert_to_parallel(layer) for layer in layers]
        super().__init__(*converted)

    def unite(self):
        """Unite all parallel layers into a single one."""
        layers = []
        for mod in self.children():
            mod_cls = mod.__class__
            if issubclass(mod_cls, HBPParallel):
                layers.append(mod.unite())
            elif issubclass(mod_cls, self.__class__):
                layers.append(mod.unite())
            else:
                layers.append(mod)
        return self.__class__(*layers)

    def split_into_blocks(self, num_blocks):
        """Split all layers into `num_blocks` parallel modules."""
        layers = []
        for mod in self.children():
            mod_cls = mod.__class__
            if issubclass(mod_cls, HBPParallel):
                layers.append(mod.split_into_blocks(num_blocks))
            elif issubclass(mod_cls, self.__class__):
                layers.append(mod.split_into_blocks(num_blocks))
            else:
                layers.append(mod)
        return self.__class__(*layers)


"""Conversions to parallel modules."""


def convert_to_parallel(layer):
    """Convert a layer to a parallelized form."""
    if issubclass(layer.__class__, HBPParallel):
        return layer

    layer_cls = layer.__class__
    if issubclass(layer_cls, HBPSequential):
        return HBPParallelSequential(*list(layer.children()))

    elif issubclass(layer_cls, HBPLinear):
        return HBPParallelLinear(layer)

    elif issubclass(layer_cls, HBPCompositionActivationLinear):
        return HBPParallelCompositionActivationLinear(layer)

    else:
        warn('Cannot convert {} to parallel module'.format(layer_cls))
        return layer
