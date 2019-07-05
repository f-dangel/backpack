from . import linear, activations, pooling, dropout, conv2d

EXTENSIONS = [
    *linear.EXTENSIONS,
    *activations.EXTENSIONS,
    *pooling.EXTENSIONS,
    *dropout.EXTENSIONS,
    *conv2d.EXTENSIONS,
]
