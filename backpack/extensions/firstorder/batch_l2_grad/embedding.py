"""BatchL2 extension for Embedding."""
from backpack.core.derivatives.embedding import EmbeddingDerivatives
from backpack.extensions.firstorder.batch_l2_grad.batch_l2_base import BatchL2Base


class BatchL2Embedding(BatchL2Base):
    """BatchL2 extension for Embedding."""

    def __init__(self):
        """Initialization."""
        super().__init__(derivatives=EmbeddingDerivatives(), params=["weight"])
