from . import (
    linear,
    activations,
    dropout,
    losses,
)

EXTENSIONS = [
    *linear.EXTENSIONS,
    *activations.EXTENSIONS,
    *dropout.EXTENSIONS,
    *losses.EXTENSIONS,
]
