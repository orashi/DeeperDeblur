import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ResBlock(nn.Module):
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

        tunnel = [ResBlock() for _ in range(len)]
        self.tunnel = nn.Sequential(*tunnel)

    def forward(self, x):
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

        up3 = nn.Sequential(nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=False),
                            nn.PixelShuffle(2),
                            nn.ReLU(inplace=True))
        up2 = nn.Sequential(nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=False),
                            nn.PixelShuffle(2),
                            nn.ReLU(inplace=True))

        self.tunnel3 = nn.Sequential(Tunnel(16),
                                     up3)
        self.tunnel2 = nn.Sequential(nn.Conv2d(128, 64, 5, 1, 2, bias=False),
                                     Tunnel(16),
                                     up2)
        self.tunnel1 = nn.Sequential(nn.Conv2d(128, 64, 5, 1, 2, bias=False),
                                     Tunnel(16))

        self.exit = nn.Sequential(nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
                                  nn.Tanh())
        # self.exit = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        for m in self.modules():  # init
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                m.weight.data = init.kaiming_normal(m.weight.data)

    def forward(self, bimg):
        bimg = self.pre(bimg)

        lv1 = self.entrance1(bimg)
        lv2 = self.entrance2(lv1)
        lv3 = self.entrance3(lv2)  # need discussion

        lv3 = self.tunnel3(lv3)
        lv2 = self.tunnel2(torch.cat([lv3, lv2.detach()], 1))
        lv1 = self.tunnel1(torch.cat([lv2, lv1.detach()], 1))

        return self.exit(lv1)
