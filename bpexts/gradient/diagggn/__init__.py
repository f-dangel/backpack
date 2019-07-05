from . import avgpool2d, maxpool2d, conv2d, linear, crossentropyloss, activations, mseloss

EXTENSIONS = [
    *avgpool2d.EXTENSIONS,
    *maxpool2d.EXTENSIONS,
    *conv2d.EXTENSIONS,
    *linear.EXTENSIONS,
    *crossentropyloss.EXTENSIONS,
    *activations.EXTENSIONS,
    *mseloss.EXTENSIONS,
]
