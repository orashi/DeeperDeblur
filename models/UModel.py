import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ResBlock(nn.Module):  # not paper original
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(64)

    def forward(self, x):
        out = self.conv1(F.relu(self.norm1(x), True))
        out = self.conv2(F.relu(self.norm2(out), True))

        out += x
        return out


class Tunnel(nn.Module):
    def __init__(self, len=16):
        super(Tunnel, self).__init__()

        # self.entrance = nn.Conv2d(in_channel, 64, kernel_size=5, stride=1, padding=2)

        tunnel = [ResBlock() for _ in range(len)]
        # tunnel += [nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)]
        self.tunnel = nn.Sequential(*tunnel)

        # self.exit = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        # x = self.entrance(x)
        # x = self.tunnel(x) + x
        # y = F.tanh(self.exit(x))

        return self.tunnel(x)


class Pyramid(nn.Module):
    def __init__(self):
        super(Pyramid, self).__init__()

        self.pre = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)

        self.entrance1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        self.entrance2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
                                       nn.ReLU(inplace=True))
        self.entrance3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        self.tunnel3 = Tunnel(16)
        self.tunnel2 = Tunnel(8)
        self.tunnel1 = Tunnel(4)

        self.up3 = nn.Sequential(nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=False),
                                 nn.PixelShuffle(2),
                                 nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=False),
                                 nn.PixelShuffle(2),
                                 nn.ReLU(inplace=True))

        self.post2 = nn.Conv2d(128, 64, 5, 1, 1, bias=False)
        self.post1 = nn.Conv2d(128, 64, 5, 1, 1, bias=False)

        for m in self.modules():  # init
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                m.weight.data = init.kaiming_normal(m.weight.data)

    def forward(self, bimg):

        lv1 = self.entrance(bimg)
        lv2 = self.entrance2(lv1)
        lv3 = self.entrance3(F.relu(lv2, True)) # need discussion

        lv3 = self.tunnel3(lv3)
        x = self.tunnel1(bimg[0])
        results.append(x[1])
        x = self.tunnel2(torch.cat([self.entrance2(bimg[1]), self.up2(x[0])], 1))
        results.append(x[1])
        x = self.tunnel3(torch.cat([self.entrance3(bimg[2]), self.up3(x[0])], 1))
        results.append(x[1])

        return
