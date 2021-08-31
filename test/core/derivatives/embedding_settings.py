"""Settings for testing derivatives of Embedding."""
from torch import randint
from torch.nn import Embedding

EMBEDDING_SETTINGS = [
    {
        "module_fn": lambda: Embedding(3, 5),
        "input_fn": lambda: randint(0, 3, (4,)),
    },
    {
        "module_fn": lambda: Embedding(5, 7),
        "input_fn": lambda: randint(0, 5, (8, 3, 3)),
    },
]
