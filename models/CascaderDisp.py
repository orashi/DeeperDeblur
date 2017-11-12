import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1):
        super(ResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=dilate, dilation=dilate,
                                   groups=cardinality,
                                   bias=False)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        if isinstance(x, tuple):
            bottleneck = self.conv_reduce.forward(torch.cat([x[0], x[1]], 1))
            bottleneck = F.relu(bottleneck, inplace=True)
            bottleneck = self.conv_conv.forward(bottleneck)
            bottleneck = F.relu(bottleneck, inplace=True)
            bottleneck = self.conv_expand.forward(bottleneck)
            return x[0] + bottleneck, x[1]
        else:
            bottleneck = self.conv_reduce.forward(x)

            bottleneck = F.relu(bottleneck, inplace=True)
            bottleneck = self.conv_conv.forward(bottleneck)
            bottleneck = F.relu(bottleneck, inplace=True)
            bottleneck = self.conv_expand.forward(bottleneck)
            return x + bottleneck


class DResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32):
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
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return residual + bottleneck


class CorrelationLayer2D(nn.Module):
    def __init__(self, max_disp=20, stride_1=1, stride_2=1):
        super(CorrelationLayer2D, self).__init__()
        self.max_displacement = max_disp
        self.stride_1 = stride_1
        self.stride_2 = stride_2

    def forward(self, x_1):
        x_1 = x_1
        x_2 = F.pad(x_1, [self.max_displacement] * 4)
        return torch.cat([torch.sum(x_1 * x_2[:, :, _x:_x + x_1.size(2), _y:_y + x_1.size(3)], 1).unsqueeze(1) for _x in
                          range(0, self.max_displacement * 2 + 1, self.stride_1) for _y in
                          range(0, self.max_displacement * 2 + 1, self.stride_2)], 1)


class Tunnel(nn.Module):
    def __init__(self, len=1, *args):
        super(Tunnel, self).__init__()

        tunnel = [DResNeXtBottleneck(*args) for _ in range(len)]
        self.tunnel = nn.Sequential(*tunnel)

    def forward(self, x):
        return self.tunnel(x)


class DilateTunnel(nn.Module):
    def __init__(self, depth=4):
        super(DilateTunnel, self).__init__()

        tunnel = [ResNeXtBottleneck(dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(dilate=8) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(dilate=1) for _ in range(14)]

        self.tunnel = nn.Sequential(*tunnel)

    def forward(self, x):
        return self.tunnel(x)


class Pyramid(nn.Module):
    def __init__(self):
        super(Pyramid, self).__init__()
        self.corr1 = nn.Sequential(CorrelationLayer2D(max_disp=20, stride_1=2, stride_2=2),
                                   nn.Conv2d(441, 32, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True))

        self.corr2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                                   nn.ReLU(inplace=True))

        self.corr3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                                   nn.ReLU(inplace=True))

        self.entrance1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
                                       nn.ReLU(inplace=True))
        self.entrance2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                                       nn.ReLU(inplace=True))
        self.entrance3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

        self.up3 = nn.Sequential(nn.Conv2d(256, 32 * 4, 3, 1, 1, bias=False),
                                 nn.PixelShuffle(2),
                                 nn.ReLU(inplace=True))
        self.con2 = nn.Conv2d(160 + 32, 32, 5, 1, 2, bias=False)

        self.up2 = nn.Sequential(nn.Conv2d(32, 32 * 4, 3, 1, 1, bias=False),
                                 nn.PixelShuffle(2),
                                 nn.ReLU(inplace=True))
        self.con1 = nn.Conv2d(96 + 32, 32, 5, 1, 2, bias=False)

        tunnel = [ResNeXtBottleneck(256 + 128, 256) for _ in range(30)]
        self.tunnel3 = nn.Sequential(*tunnel)

        depth = 3
        tunnel = [ResNeXtBottleneck(64 + 64, 32, cardinality=8, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(64 + 64, 32, cardinality=8, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(64 + 64, 32, cardinality=8, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(64 + 64, 32, cardinality=8, dilate=2),
                   ResNeXtBottleneck(64 + 64, 32, cardinality=8, dilate=1)]
        self.tunnel2 = nn.Sequential(*tunnel)

        tunnel = [ResNeXtBottleneck(64 + 32, 32, cardinality=8, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(64 + 32, 32, cardinality=8, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(64 + 32, 32, cardinality=8, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(64 + 32, 32, cardinality=8, dilate=8) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(64 + 32, 32, cardinality=8, dilate=2),
                   ResNeXtBottleneck(64 + 32, 32, cardinality=8, dilate=1)]
        self.tunnel1 = nn.Sequential(*tunnel)

        self.exit = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        corr1 = self.corr1(x)
        corr2 = self.corr2(corr1)
        corr3 = self.corr3(corr2)

        lv1 = self.entrance1(x)
        lv2 = self.entrance2(lv1)
        lv3 = self.entrance3(lv2)  # need discussion

        lv3_up = self.up3(lv3)
        lv3 = self.up3(self.tunnel3((lv3, corr3))[0])

        lv2_up = self.up2(lv3)
        lv2 = self.tunnel2((self.con2(torch.cat([lv2, lv3, lv3_up], 1)), torch.cat([lv3, corr2], 1)))[0] + lv3
        lv2 = self.up2(lv2)

        lv1 = self.tunnel1((self.con1(torch.cat([lv1, lv2, lv2_up], 1)), torch.cat([lv2, corr1], 1)))[0] + lv2
        return self.exit(lv1)


class PatchD(nn.Module):
    def __init__(self, ndf=64):
        super(PatchD, self).__init__()

        sequence = [
            nn.Conv2d(6, ndf, kernel_size=4, stride=2, padding=1, bias=False),  # 128
            nn.LeakyReLU(0.2, True),

            Tunnel(2, ndf, ndf),
            DResNeXtBottleneck(ndf, ndf * 2, 2),  # 64

            Tunnel(3, ndf * 2, ndf * 2),
            DResNeXtBottleneck(ndf * 2, ndf * 4, 2),  # 32

            Tunnel(4, ndf * 4, ndf * 4),
            DResNeXtBottleneck(ndf * 4, ndf * 8, 2),  # 16

            Tunnel(4, ndf * 8, ndf * 8),
            DResNeXtBottleneck(ndf * 8, ndf * 16, 2),  # 8

            Tunnel(2, ndf * 16, ndf * 16),
            DResNeXtBottleneck(ndf * 16, ndf * 32, 2),  # 4

            nn.Conv2d(ndf * 32, 1, kernel_size=4, stride=1, padding=0, bias=False)

        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

        # TODO: fix relu bug
