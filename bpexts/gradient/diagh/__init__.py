from . import conv2d, linear, sigmoid, tanh, relu, crossentropyloss, mseloss, pooling, dropout

EXTENSIONS = [
    *conv2d.EXTENSIONS,
    *linear.EXTENSIONS,
    *crossentropyloss.EXTENSIONS,
    *mseloss.EXTENSIONS,
    *sigmoid.EXTENSIONS,
    *relu.EXTENSIONS,
    *tanh.EXTENSIONS,
    *dropout.EXTENSIONS,
    *pooling.EXTENSIONS,
]
