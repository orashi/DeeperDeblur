import torch.nn as nn
import torch.nn.functional as F


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
        """
        super(ResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(bottleneck, inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(bottleneck, inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return residual + bottleneck


class DResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
        """
        super(DResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return residual + bottleneck


class Tunnel(nn.Module):
    def __init__(self, len=1, bottleneck=ResNeXtBottleneck, *args):
        super(Tunnel, self).__init__()

        tunnel = [bottleneck(*args) for _ in range(len)]
        self.tunnel = nn.Sequential(*tunnel)

    def forward(self, x):
        return self.tunnel(x)


class Pyramid(nn.Module):
    def __init__(self):
        super(Pyramid, self).__init__()

        self.entrance = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=4, bias=False),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                      nn.ReLU(inplace=True)
                                      )

        self.tunnel = nn.Sequential(Tunnel(60))

        self.exit = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, bias=False),
                                  nn.PixelShuffle(2),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)
                                  )

    def forward(self, x):
        x = self.entrance(x)
        x = self.tunnel(x)
        return self.exit(x)


class PatchD(nn.Module):
    def __init__(self, ndf=64):
        super(PatchD, self).__init__()

        sequence = [
            nn.Conv2d(6, ndf, kernel_size=4, stride=2, padding=1, bias=False),  # 128
            nn.LeakyReLU(0.2, True),

            Tunnel(2, DResNeXtBottleneck, ndf, ndf),
            ResNeXtBottleneck(ndf, ndf * 2, 2),  # 64

            Tunnel(3, DResNeXtBottleneck, ndf * 2, ndf * 2),
            ResNeXtBottleneck(ndf * 2, ndf * 4, 2),  # 32

            Tunnel(4, DResNeXtBottleneck, ndf * 4, ndf * 4),
            ResNeXtBottleneck(ndf * 4, ndf * 8, 2),  # 16

            Tunnel(4, DResNeXtBottleneck, ndf * 8, ndf * 8),
            ResNeXtBottleneck(ndf * 8, ndf * 16, 2),  # 8

            Tunnel(2, DResNeXtBottleneck, ndf * 16, ndf * 16),
            ResNeXtBottleneck(ndf * 16, ndf * 32, 2),  # 4

            nn.Conv2d(ndf * 32, 1, kernel_size=4, stride=1, padding=0, bias=False)

        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

        # TODO: fix relu bug
