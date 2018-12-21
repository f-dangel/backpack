"""Parallel series of layers of same class."""

from .parallel import HBPParallel
from ..linear import HBPLinear
from ..combined import HBPCompositionActivationLinear
from .converter_linear import HBPParallelConverterLinear
from .converter_combined import HBPParallelConverterCompositionActivationLinear


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

    def split_into_blocks(self, num_blocks):
        """Split layer into `num_blocks` parallel modules."""
        out_features_list = self.compute_out_features_list(num_blocks)
        return self.split(out_features_list)

    def get_converter(self):
        """Return the appropriate converter for layers."""
        if self.layer_class is HBPLinear:
            return HBPParallelConverterLinear
        elif issubclass(self.layer_class, HBPCompositionActivationLinear):
            return HBPParallelConverterCompositionActivationLinear
        else:
            raise ValueError('No conversion known for layer of type '
                             '{}'.format(self.layer_class))

    def total_out_features(self):
        """Return the number of out_features in total."""
        if self.layer_class is HBPLinear:
            return sum(mod.out_features for mod in self.children())
        elif issubclass(self.layer_class, HBPCompositionActivationLinear):
            return sum(mod.linear.out_features for mod in self.children())
        else:
            raise ValueError('No method for getting outputs known for layer '
                             '{}'.format(self.layer_class))

    def compute_out_features_list(self, num_blocks):
        """Compute the sizes of the output when splitting into blocks."""
        out_features = self.total_out_features()
        if num_blocks <= 0:
            raise ValueError('Parameter splitting only valid for'
                             ' non-negative number of blocks, but '
                             ' got {}'.format(num_blocks))
        num_blocks = min(out_features, num_blocks)
        block_size, block_rest = divmod(out_features, num_blocks)
        out_features_list = num_blocks * [block_size]
        if block_rest != 0:
            for i in range(block_rest):
                out_features_list[i] += 1
        return list(out_features_list)
