import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ResBlock(nn.Module):  # paper original
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        out = F.relu(self.conv1(x), True)
        out = self.conv2(out)

        out += x
        return out


class Tunnel(nn.Module):
    def __init__(self, in_channel, ngf=64, len=19):
        super(Tunnel, self).__init__()

        self.entrance = nn.Conv2d(in_channel, ngf, kernel_size=5, stride=1, padding=2)

        tunnel = [ResBlock() for _ in range(len)]
        self.tunnel = nn.Sequential(*tunnel)

        self.exit = nn.Conv2d(ngf, 3, kernel_size=5, stride=1, padding=2)  # no tanh???????

    def forward(self, x):
        x = self.entrance(x)
        x = self.tunnel(x)
        x = self.exit(x)

        return x


class Pyramid(nn.Module):
    def __init__(self):
        super(Pyramid, self).__init__()

        self.tunnel1 = Tunnel(3)
        self.tunnel2 = Tunnel(6)
        self.tunnel3 = Tunnel(6)

        self.up1 = nn.Sequential(nn.Conv2d(3, 3 * 4, 5, 1, 2, bias=False),
                                 nn.PixelShuffle(2))
        self.up2 = nn.Sequential(nn.Conv2d(3, 3 * 4, 5, 1, 2, bias=False),
                                 nn.PixelShuffle(2))

        for m in self.modules():  # init
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                m.weight.data = init.kaiming_normal(m.weight.data)

    def forward(self, *bimg):
        results = []

        x = self.tunnel1(bimg[0])
        results.append(x)
        x = self.tunnel2(torch.cat([bimg[1], self.up1(x)], 1))
        results.append(x)
        x = self.tunnel3(torch.cat([bimg[2], self.up2(x)], 1))
        results.append(x)

        return results
