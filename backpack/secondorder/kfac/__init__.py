from . import pooling, activations, linear, conv2d, dropout, losses, padding

EXTENSIONS = [
    *pooling.EXTENSIONS,
    *activations.EXTENSIONS,
    *linear.EXTENSIONS,
    *dropout.EXTENSIONS,
    *conv2d.EXTENSIONS,
    *losses.EXTENSIONS,
    *padding.EXTENSIONS,
]
