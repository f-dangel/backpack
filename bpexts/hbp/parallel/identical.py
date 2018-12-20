"""Parallel series of layers of same class."""

from .parallel import HBPParallel
from ..linear import HBPLinear
from .converter_linear import HBPParallelConverterLinear


class HBPParallelIdentical(HBPParallel):
    """Multiple layers in parallel, all of the same type.

    Perform conversion from a usual HBP module to a parallel one,
    take care of parameter splitting and uniting.
    """
    def __init__(self, layers):
        different_classes = set([l.__class__ for l in layers])
        if len(different_classes) != 1:
            raise ValueError('Expecting layers of same type,'
                             ' got {}'.format(different_classes))
        self.layer_class = different_classes.pop()
        super().__init__(layers)

    @classmethod
    def from_module(cls, layer):
        """Convert HBPModule to HBPParallel with a single submodule.

        Parameters:
        -----------
        layer : (HBPModule)
            Module to be converted to a parallel series

        Returns:
        --------
        (HBPParallel)
            Parallel series consisting with only one submodule
        """
        return cls([layer])

    def unite(self):
        """Unite all parallel layers into a single one."""
        converter = self.get_converter()
        layer = converter.unite(self)
        return self.__class__([layer])

    def split(self, out_features_list):
        """Split layer into multiple parallel ones."""
        united = self.unite()
        converter = self.get_converter()
        layers = converter.split(united, out_features_list)
        return self.__class__(layers)

    def get_converter(self):
        """Return the appropriate converter for layers."""
        if self.layer_class is HBPLinear:
            return HBPParallelConverterLinear
        else:
            raise ValueError('No conversion known for layer of type '
                             '{}'.format(self.layer_class))
