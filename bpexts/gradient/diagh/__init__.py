from . import conv2d, linear, sigmoid, tanh, relu, crossentropyloss, mseloss, maxpool2d

EXTENSIONS = conv2d.EXTENSIONS + linear.EXTENSIONS + crossentropyloss.EXTENSIONS + mseloss.EXTENSIONS + sigmoid.EXTENSIONS + relu.EXTENSIONS + tanh.EXTENSIONS + maxpool2d.EXTENSIONS
