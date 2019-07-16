from . import linear, activations, pooling, dropout, conv2d, losses, padding

EXTENSIONS = [
    *linear.EXTENSIONS,
    *activations.EXTENSIONS,
    *pooling.EXTENSIONS,
    *dropout.EXTENSIONS,
    *conv2d.EXTENSIONS,
    *losses.EXTENSIONS,
    *padding.EXTENSIONS,
]
