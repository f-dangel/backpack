"""Support parameter splitting for a sequence of parallel modules."""

from ..sequential import HBPSequential
from .parallel import HBPParallel


class HBPParallelSequential(HBPSequential):
    """Handle splitting for a sequence of parallel identical modules."""
    @classmethod
    def from_sequential(cls, sequence):
        """Convert from HBPSequential.

        Parameters:
        -----------
        sequence : (HBPSequential)
            Sequence of modules that all allow parameter splitting.
        """
        layers = [HBPParallel.from_module(mod)
                  for mod in sequence.children()]
        return cls(*layers)

    def unite(self):
        """Unite all parallel layers into a single one."""
        layers = [mod.unite() for mod in self.children()]
        return self.__class__(*layers)

    # override
    def split(self, out_features_list):
        """Split layer into multiple parallel ones.

        As the outputs may vary from layer to layer in the
        sequence, this is not supported.
        """
        raise NotImplementedError('Not supported, you have to split'
                                  ' all constituents manually')

    def split_into_blocks(self, num_blocks):
        """Split all layers into `num_blocks` parallel modules."""
        layers = [mod.split_into_blocks(num_blocks)
                  for mod in self.children()]
        return self.__class__(*layers)
