from . import conv2d, linear, crossentropyloss, mseloss

EXTENSIONS = conv2d.EXTENSIONS + linear.EXTENSIONS + crossentropyloss.EXTENSIONS + mseloss.EXTENSIONS
