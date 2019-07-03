from . import avgpool2d, maxpool2d, conv2d, linear, crossentropyloss, tanh, relu, sigmoid, mseloss

EXTENSIONS = [
    *avgpool2d.EXTENSIONS,
    *maxpool2d.EXTENSIONS,
    *conv2d.EXTENSIONS,
    *linear.EXTENSIONS,
    *crossentropyloss.EXTENSIONS,
    *tanh.EXTENSIONS,
    *relu.EXTENSIONS,
    *sigmoid.EXTENSIONS,
    *mseloss.EXTENSIONS,
]
