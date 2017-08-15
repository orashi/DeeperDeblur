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

        self.up2 = nn.Sequential(nn.Conv2d(3, 3 * 4, 5, 1, 2),
                                 nn.PixelShuffle(2))
        self.up3 = nn.Sequential(nn.Conv2d(3, 3 * 4, 5, 1, 2),
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
        x = self.tunnel3(torch.cat([bimg[2], self.up3(x)], 1))
        results.append(x)

        return results


class DisBlock(nn.Module):  # not paper original
    def __init__(self, inc, ouc, str, filtSize=5, pad=2):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(inc, ouc, kernel_size=filtSize, stride=str, padding=pad, bias=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv(x), 0.2, True)
        return out


class Discriminator(nn.Module):
    def __init__(self, ndf=32):
        super(Discriminator, self).__init__()

        sequence = [
            DisBlock(3, ndf, 1),
            DisBlock(ndf, ndf, 2),          # 128
            DisBlock(ndf, ndf * 2, 1),
            DisBlock(ndf * 2, ndf * 2, 2),  # 64
            DisBlock(ndf * 2, ndf * 4, 1),
            DisBlock(ndf * 4, ndf * 4, 4),  # 16
            DisBlock(ndf * 4, ndf * 8, 1),
            DisBlock(ndf * 8, ndf * 8, 4),  # 4
            DisBlock(ndf * 8, ndf * 16, 1),
            DisBlock(ndf * 16, ndf * 16, 4, filtSize=4, pad=0),  # 1
        ]

        self.down = nn.Sequential(*sequence)

        self.linear = nn.Linear(ndf * 16, 1)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data = init.kaiming_normal(m.weight.data, a=0.2)
            elif classname.find('Linear') != -1:
                m.weight.data = init.kaiming_normal(m.weight.data, a=0.2)

    def forward(self, input):
        x = self.down(input)
        x = F.sigmoid(self.linear(x.view(-1)))
        return x


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

# TODO: discuss about affine and bias
