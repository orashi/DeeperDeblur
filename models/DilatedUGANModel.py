import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ResBlock(nn.Module):
    def __init__(self, Dilation=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=Dilation, dilation=Dilation)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=Dilation, dilation=Dilation)

    def forward(self, x):
        out = self.conv1(F.relu(x, True))
        out = self.conv2(F.relu(out, True))

        out += x
        return out


class Block(nn.Module):
    def __init__(self, Dilation=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=Dilation, dilation=Dilation)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=Dilation, dilation=Dilation)

    def forward(self, x):
        out = self.conv1(F.relu(x, True))
        out = self.conv2(F.relu(out, True))

        return out


class Tunnel(nn.Module):
    def __init__(self, len=1):
        super(Tunnel, self).__init__()

        tunnel = [ResBlock() for _ in range(len)]
        self.tunnel = nn.Sequential(*tunnel)

    def forward(self, x):
        return self.tunnel(x)


class DilateTunnel(nn.Module):
    def __init__(self, depth=3):
        super(DilateTunnel, self).__init__()

        tunnel = [ResBlock(1) for _ in range(depth)]
        tunnel += [ResBlock(2) for _ in range(depth)]
        tunnel += [ResBlock(4) for _ in range(depth)]
        tunnel += [Block(2), Block(1)]
        self.tunnel = nn.Sequential(*tunnel)

    def forward(self, x):
        return self.tunnel(x)

class DilateTunnel2(nn.Module):
    def __init__(self, depth=2):
        super(DilateTunnel2, self).__init__()

        tunnel = [ResBlock(1) for _ in range(depth)]
        tunnel += [ResBlock(2) for _ in range(depth)]
        tunnel += [ResBlock(4) for _ in range(depth)]
        tunnel += [ResBlock(8) for _ in range(depth)]
        tunnel += [Block(2), Block(1)]
        self.tunnel = nn.Sequential(*tunnel)

    def forward(self, x):
        return self.tunnel(x)


class Pyramid(nn.Module):
    def __init__(self):
        super(Pyramid, self).__init__()

        self.entrance1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
                                       nn.ReLU(inplace=True))
        self.entrance2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
                                       nn.ReLU(inplace=True))
        self.entrance3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        self.up3 = nn.Sequential(nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=False),
                                 nn.PixelShuffle(2),
                                 nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=False),
                                 nn.PixelShuffle(2),
                                 nn.ReLU(inplace=True))

        self.tunnel3 = nn.Sequential(Tunnel(20))
        self.tunnel2 = nn.Sequential(nn.Conv2d(128, 64, 5, 1, 2, bias=False),
                                     DilateTunnel(3))
        self.tunnel1 = nn.Sequential(nn.Conv2d(128, 64, 5, 1, 2, bias=False),
                                     DilateTunnel2(2))

        self.exit = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, bimg):
        lv1 = self.entrance1(bimg)
        lv2 = self.entrance2(lv1)
        lv3 = self.entrance3(lv2)  # need discussion

        results = []
        lv3 = self.tunnel3(lv3)
        results.append(self.exit(lv3))

        up3 = self.up3(lv3)
        lv2 = self.tunnel2(torch.cat([up3, lv2.detach()], 1)) + up3
        results.append(self.exit(lv2))

        up2 = self.up2(lv2)
        lv1 = self.tunnel1(torch.cat([up2, lv1.detach()], 1)) + up2
        results.append(self.exit(lv1))

        return results


class PatchD(nn.Module):
    def __init__(self, ndf=64):
        super(PatchD, self).__init__()

        sequence = [
            nn.Conv2d(6, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * 1, ndf * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * 2, ndf * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * 4, ndf * 8,
                      kernel_size=4, stride=1, padding=1),  # stride 1
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.loss = nn.MSELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

# TODO: fix relu bug