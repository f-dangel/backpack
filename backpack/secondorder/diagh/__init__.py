from . import conv2d, linear, losses, pooling, dropout, activations, padding

EXTENSIONS = [
    *conv2d.EXTENSIONS,
    *linear.EXTENSIONS,
    *losses.EXTENSIONS,
    *activations.EXTENSIONS,
    *dropout.EXTENSIONS,
    *pooling.EXTENSIONS,
    *padding.EXTENSIONS,
]
