from . import pooling, conv2d, linear, crossentropyloss, activations, mseloss

EXTENSIONS = [
    *pooling.EXTENSIONS,
    *conv2d.EXTENSIONS,
    *linear.EXTENSIONS,
    *crossentropyloss.EXTENSIONS,
    *activations.EXTENSIONS,
    *mseloss.EXTENSIONS,
]
