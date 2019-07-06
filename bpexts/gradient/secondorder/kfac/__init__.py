from . import pooling, activations, linear, conv2d, dropout

EXTENSIONS = [
    *pooling.EXTENSIONS,
    *activations.EXTENSIONS,
    *linear.EXTENSIONS,
    *dropout.EXTENSIONS,
    *conv2d.EXTENSIONS,
]
