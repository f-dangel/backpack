from . import pooling, conv2d, linear, activations, losses, padding

EXTENSIONS = [
    *pooling.EXTENSIONS,
    *conv2d.EXTENSIONS,
    *linear.EXTENSIONS,
    *activations.EXTENSIONS,
    *losses.EXTENSIONS,
    *padding.EXTENSIONS,
]
