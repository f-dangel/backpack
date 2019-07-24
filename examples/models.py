import torch
from torch.nn import ReLU, Conv2d, Linear, Dropout, Sequential
from utils import tfconv2d, tfmaxpool2d
from backpack.core.layers import Flatten


class VGGSequentialBase(Sequential):
    VARIANTS = ['16', '19']
    REQUIRES_ADDITIONAL_CONV = '19'
    IN_CHANNELS = 3
    CHANNELS = [64, 128, 256, 512, 512]
    CONV_FEATURES = 25088
    LIN_FEATURES = 4096

    def __init__(self, num_outputs, variant='16'):
        """Assume input images of dimension 3x224x224."""
        self.check_variant(variant)
        self.variant = variant
        self.num_outputs = num_outputs

        super().__init__()

        features = Sequential(
            *self.conv1(),
            *self.conv2(),
            *self.conv3(),
            *self.conv4(),
            *self.conv5(),
        )
        self.add_module('features', features)

        self.add_module('flatten', Flatten())

        classifier = Sequential(
            *self.fc1(),
            *self.fc2(),
            *self.fc3(),
        )
        self.add_module('classifier', classifier)

    def conv1(self):
        prev_channels = self.IN_CHANNELS
        channels = self.CHANNELS[0]

        return [
            *self.conv_same_with_act(prev_channels, channels),
            *self.conv_same_with_act(channels, channels),
            self.maxpool_same_layer(),
        ]

    def conv2(self):
        prev_channels = self.CHANNELS[0]
        channels = self.CHANNELS[1]

        return [
            *self.conv_same_with_act(prev_channels, channels),
            *self.conv_same_with_act(channels, channels),
            self.maxpool_same_layer(),
        ]

    def conv3(self):
        prev_channels = self.CHANNELS[1]
        channels = self.CHANNELS[2]

        modules = [
            *self.conv_same_with_act(prev_channels, channels),
            *self.conv_same_with_act(channels, channels),
            *self.conv_same_with_act(channels, channels),
        ]

        if self.require_additional_conv():
            modules += self.conv_same_with_act(channels, channels)

        modules.append(self.maxpool_same_layer())
        return modules

    def conv4(self):
        prev_channels = self.CHANNELS[2]
        channels = self.CHANNELS[3]

        modules = [
            *self.conv_same_with_act(prev_channels, channels),
            *self.conv_same_with_act(channels, channels),
            *self.conv_same_with_act(channels, channels),
        ]

        if self.require_additional_conv():
            modules += self.conv_same_with_act(channels, channels)

        modules.append(self.maxpool_same_layer())
        return modules

    def conv5(self):
        prev_channels = self.CHANNELS[3]
        channels = self.CHANNELS[4]

        modules = [
            *self.conv_same_with_act(prev_channels, channels),
            *self.conv_same_with_act(channels, channels),
            *self.conv_same_with_act(channels, channels),
        ]

        if self.require_additional_conv():
            modules += self.conv_same_with_act(channels, channels)

        modules.append(self.maxpool_same_layer())
        return modules

    def fc1(self):
        return [
            Linear(self.CONV_FEATURES, self.LIN_FEATURES),
            self.activation(),
            Dropout(p=0.5),
        ]

    def fc2(self):
        return [
            Linear(self.LIN_FEATURES, self.LIN_FEATURES),
            self.activation(),
            Dropout(p=0.5),
        ]

    def fc3(self):
        return [
            Linear(self.LIN_FEATURES, self.num_outputs),
        ]

    def activation(self):
        return ReLU()

    def conv_same_with_act(self,
                           in_channels,
                           out_channels,
                           kernel_size=3,
                           stride=1):
        return [
            tfconv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                tf_padding_type='same'),
            self.activation(),
        ]

    def maxpool_same_layer(self, kernel_size=2, stride=2):
        return tfmaxpool2d(kernel_size, stride=stride, tf_padding_type='same')

    def check_variant(self, variant):
        if not variant in self.VARIANTS:
            raise ValueError("Known variants: {}".format(self.VARIANTS))

    def require_additional_conv(self):
        return self.variant is self.REQUIRES_ADDITIONAL_CONV


class VGG16Sequential(VGGSequentialBase):
    def __init__(self, num_outputs):
        super().__init__(num_outputs, variant='16')


class VGG19Sequential(VGGSequentialBase):
    def __init__(self, num_outputs):
        super().__init__(num_outputs, variant='19')
