from . import maxpool2d, conv2d, linear, crossentropyloss, relu, sigmoid, mseloss

SIGNATURE = (maxpool2d.SIGNATURE + conv2d.SIGNATURE + linear.SIGNATURE +
             crossentropyloss.SIGNATURE + relu.SIGNATURE + sigmoid.SIGNATURE +
             mseloss.SIGNATURE)
