import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d


def initialize_fc_weights(f):
    for m in f.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def make_fc_block(in_channels, v, input_shape=None, **kwargs):
    layers = []
    layers += [nn.Linear(in_channels, v)]
    layers += [nn.BatchNorm1d(v)]
    if 'nonlinearity' in kwargs:
        layers += [kwargs['nonlinearity']]
    else:
        layers += [nn.ReLU(inplace=True)]
    return layers, (v, )


def initialize_vgg_weights(f):
    for m in f.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def make_vgg_block(in_channels, v, input_shape=None, **kwargs):
    layers = []
    layers += [nn.ReplicationPad2d(1)]
    layers += [nn.Conv2d(in_channels, v, kernel_size=3)]
    layers += [nn.BatchNorm2d(v)]
    if 'nonlinearity' in kwargs:
        layers += [kwargs['nonlinearity']]
    else:
        layers += [nn.ReLU(inplace=True)]

    if input_shape:
        output_shape = (v, input_shape[1], input_shape[2])
    else:
        output_shape = None

    return layers, output_shape


class ResnetBlock(nn.Module):
    def __init__(self, in_planes, planes, nonlinearity, **kwargs):
        super(ResnetBlock, self).__init__()
        self.p1 = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.nonlin1 = nonlinearity
        self.p2 = nn.ReplicationPad2d(1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.nonlin2 = nonlinearity

    def forward(self, x):
        out = self.nonlin1(self.bn1(self.conv1(self.p1(x))))
        out = self.bn2(self.conv2(self.p2(out)))

        # pad the image if its size is not divisible by 2
        padding_h = 0 if x.size(2) % 2 == 0 else 1
        padding_w = 0 if x.size(3) % 2 == 0 else 1
        id = avg_pool2d(x, 1, stride=1, padding=(padding_h, padding_w))

        # this assumes we are always doubling the amount of kernels as we go deeper
        if id.size(1) < out.size(1):
            id = torch.cat((id, id), dim=1)

        # if less channels in next layer, then halve
        elif id.size(1) > out.size(1):
            id = torch.add(*id.chunk(2, dim=1)) / 2.0

        return self.nonlin2(out + id)


def initialize_resnet_weights(f):
    for m in f.modules():
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def make_resnet_block(in_channels, out_channels, input_shape=None, **kwargs):
    layers = []
    layers += [ResnetBlock(in_channels, out_channels, **kwargs)]

    if input_shape:
        output_shape = (out_channels, *conv_output_shape(input_shape[1:3], kernel_size=3, stride=1, pad=1))
    else:
        output_shape = None
    return layers, output_shape


class FixupResLayer(nn.Module):
    def __init__(self, in_layers, filters, stride=1, nonlinearity=None):
        super().__init__()
        self.c1 = nn.Conv2d(in_layers, filters, 3, stride=stride, padding=1, bias=False)
        self.c2 = nn.Conv2d(filters, filters, 3, stride=1, padding=1, bias=False)
        self.c2.weight.data.zero_()
        self.stride = stride
        self.nonlin1 = nonlinearity
        self.nonlin2 = nonlinearity

        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(4)])

    def forward(self, input):
        hidden = input + self.bias[0]
        hidden = self.c1(hidden) + self.bias[1]
        hidden = self.nonlin1(hidden) + self.bias[2]
        hidden = self.c2(hidden) * self.gain + self.bias[3]

        # pad the image if its size is not divisible by 2
        padding_h = 0 if input.size(2) % 2 == 0 else 1
        padding_w = 0 if input.size(3) % 2 == 0 else 1
        id = avg_pool2d(input, self.stride, stride=self.stride, padding=(padding_h, padding_w))

        # if more channels in the next layer, then double
        if id.size(1) < hidden.size(1):
            id = torch.cat((id, id), dim=1)

        # if less channels in next layer, then halve
        elif id.size(1) > hidden.size(1):
            id = torch.add(*id.chunk(2, dim=1)) / 2.0

        return self.nonlin2(hidden + id)


def initialize_resnet_fixup_weights(f):
    depth = 1
    for module in f:
        if isinstance(module, nn.Conv2d):
            depth += 1
        if isinstance(module, FixupResLayer):
            module.c1.weight.data.mul_(depth ** -0.5)
            depth += 1


def get_arg(kwargs, arg, default=None):
    if arg in kwargs:
        return kwargs[arg]
    else:
        return default


def make_resnet_fixup_block(in_channels, out_channels, input_shape=None, **kwargs):
    nonlin = get_arg(kwargs, 'nonlinearity', default=nn.ReLU(inplace=True))
    stride = get_arg(kwargs, 'stride', default=1)
    layers = []
    layers += [FixupResLayer(in_channels, out_channels, stride=stride, nonlinearity=nonlin)]
    if input_shape is not None:
        output_shape = out_channels, *conv_output_shape(input_shape[1:3], kernel_size=3, stride=stride, pad=1)
    else:
        output_shape = None
    return layers, output_shape


class NetworkType:
    def __init__(self, make_block, initialize_weights):
        self.make_block = make_block
        self.initialize_weights = initialize_weights


network_types = {
    'fc': NetworkType(make_fc_block, initialize_fc_weights),
    'vgg': NetworkType(make_vgg_block, initialize_vgg_weights),
    'resnet-batchnorm': NetworkType(make_resnet_block, initialize_resnet_weights),
    'resnet-fixup': NetworkType(make_resnet_fixup_block, initialize_resnet_fixup_weights)
}


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    from math import floor
    if type(kernel_size) is int:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


def scan_token(token):
    t = token.split(':')
    if len(t) == 3:
        return t[0], int(t[1]), int(t[2])
    if len(t) == 1:
        return t[0], None, None
    raise Exception('Token format is either str, or str:int:int')


def make_layers(type, cfg, input_shape=None, nonlinearity=None, init_weights=True, **kwargs):
    """

    :param type: the network type, can be 'fc', 'vgg', 'resnet-batchnorm', 'resnet-fixup'
    :param cfg: list of strings, each string is a token, can be..
      B:<input_channels>:<output_channels>
      C:<input_channels>:<output_channels>
      M: Max pooling
      U: Bilinear upsample
    :param input_shape:  optional: (C, H, W) tuple indicating the data input shape
    :param nonlinearity:  optional: nonlinearity to use in the network
    :param init_weights:  optional: initialize the weights
    :param kwargs:  kwargs specific for the block type
    :return:
      backbone:  the network
      shape: optional:  a list of [(C, H, W), (C, H, W) ... ] of the shapes activations at each block output, requires
        input_shape to be set, or it will return a list of None
    """

    layers = []
    shapes = [input_shape]
    nonlinearity = nn.ReLU(inplace=True) if nonlinearity is None else nonlinearity

    for token in cfg:
        block_type, in_channels, out_channels = scan_token(token)

        if block_type == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            if input_shape:  # compute output shape
                shapes.append((shapes[-1][0], *conv_output_shape(shapes[-1][1:3], kernel_size=2, stride=2)))
                if min(*shapes[-1][1:3]) <= 0:
                    raise Exception('Image downsampled to 0 or less, use less downsampling')

        elif block_type == 'U':
            layers += [nn.UpsamplingBilinear2d(scale_factor=2)]

            if input_shape:  # compute output shape
                shapes.append((shapes[-1][0], shapes[-1][1] * 2, shapes[-1][2] * 2))

        elif block_type == 'C':
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)]
            layers += [nonlinearity]

            if input_shape:  # compute output shape
                shapes.append((out_channels, *conv_output_shape(shapes[-1][1:3], kernel_size=3, stride=1, pad=1)))

        elif block_type == 'B':
            block, output_shape = network_types[type].make_block(in_channels, out_channels, shapes[-1],
                                                                 nonlinearity=nonlinearity, **kwargs)
            layers += block
            if input_shape:
                shapes.append(output_shape)

    layer = nn.Sequential(*layers)

    if init_weights:
        network_types[type].initialize_weights(layer)
    return layer, shapes


