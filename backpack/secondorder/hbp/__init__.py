from . import losses, linear, activations, dropout

EXTENSIONS = [
    *losses.EXTENSIONS,
    *linear.EXTENSIONS,
    *activations.EXTENSIONS,
    *dropout.EXTENSIONS,
]
