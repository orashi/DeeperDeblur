import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ResBlock(nn.Module):  # not paper original
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        # self.norm1 = nn.InstanceNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        # self.norm2 = nn.InstanceNorm2d(64)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(F.relu(out, True))
        out += x
        return out


class Tunnel(nn.Module):
    def __init__(self, in_channel, len=19):
        super(Tunnel, self).__init__()

        tunnel = [nn.Conv2d(in_channel, 64, kernel_size=5, stride=1, padding=2)]
        tunnel += [ResBlock() for _ in range(len)]
        tunnel += [nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2)]
        self.tunnel = nn.Sequential(*tunnel)

    def forward(self, x):
        return self.tunnel(x)


class Pyramid(nn.Module):
    def __init__(self):
        super(Pyramid, self).__init__()

        self.tunnel1 = Tunnel(3, 19)
        self.tunnel2 = Tunnel(6, 19)
        self.tunnel3 = Tunnel(6, 19)

        self.up2 = nn.Sequential(nn.Conv2d(3, 3 * 4, 3, 1, 1, bias=False),
                                 nn.PixelShuffle(2))
        self.up3 = nn.Sequential(nn.Conv2d(3, 3 * 4, 3, 1, 1, bias=False),
                                 nn.PixelShuffle(2))

        for m in self.modules():  # init
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                m.weight.data = init.kaiming_normal(m.weight.data)

    def forward(self, *bimg):
        results = []

        x = self.tunnel1(bimg[0])
        results.append(x)
        x = self.tunnel2(torch.cat([bimg[1], self.up2(x)], 1))
        results.append(x)
        x = self.tunnel3(torch.cat([self.entrance3(bimg[2]), self.up3(x)], 1))
        results.append(x)

        return results
