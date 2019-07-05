from . import linear, activations, pooling, dropout

EXTENSIONS = [
    *linear.EXTENSIONS,
    *activations.EXTENSIONS,
    *pooling.EXTENSIONS,
    *dropout.EXTENSIONS,
]
