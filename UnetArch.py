import argparse
import os
import random

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.utils as vutils
from torch.autograd import Variable
from visdom import Visdom

from data.UData import CreateDataLoader
from models.UModel import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--cut', type=int, default=2, help='cut backup frequency')
parser.add_argument('--niter', type=int, default=700, help='number of epochs to train for')
parser.add_argument('--ngf', type=int, default=64)
# parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate, default=0.0001')
# parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
# parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
# parser.add_argument('--Diters', type=int, default=1, help='number of D iters per each G iter')
parser.add_argument('--manualSeed', type=int, default=2345, help='random seed to use. Default=1234')
# parser.add_argument('--baseGeni', type=int, default=2500, help='start base of pure pair L1 loss')
parser.add_argument('--geni', type=int, default=0, help='continue gen image num')
parser.add_argument('--epoi', type=int, default=0, help='continue epoch num')
parser.add_argument('--env', type=str, default='main', help='visdom env')
# parser.add_argument('--gpW', type=float, default=10, help='gradient penalty weight')

opt = parser.parse_args()
print(opt)

####### regular set up
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
gen_iterations = opt.geni
try:
    os.makedirs(opt.outf)
except OSError:
    pass
# random seed setup                                  # !!!!!
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
cudnn.benchmark = True
####### regular set up end


viz = Visdom(env=opt.env)

dataloader = CreateDataLoader(opt)

netG = Pyramid()
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

criterion_L1 = nn.L1Loss()
criterion_L2 = nn.MSELoss()

if opt.cuda:
    netG.cuda()
    criterion_L1.cuda()
    criterion_L2.cuda()

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.9))
scheduler = lr_scheduler.MultiStepLR(optimizerG, milestones=[30], gamma=0.1)  # 1.5*10^5 iter

flag = 1
flag2 = 1
flag3 = 1
for epoch in range(opt.epoi, opt.niter):
    scheduler.step()
    data_iter = iter(dataloader)
    iter_count = 0
    while iter_count < len(dataloader):
        ############################
        # Update G network
        ############################
        netG.zero_grad()

        data = data_iter.next()
        iter_count += 1

        if opt.cuda:
            data = list(map(lambda x: x.cuda(), data))

        real_bim, real_sim = data[0], data[1]

        if flag:  # fix samples
            viz.images(
                real_bim.mul(0.5).add(0.5).cpu().numpy(),
                opts=dict(title='blur img', caption='level final')
            )
            viz.images(
                real_sim.mul(0.5).add(0.5).cpu().numpy(),
                opts=dict(title='sharp img', caption='level final')
            )

            vutils.save_image(real_bim.mul(0.5).add(0.5),
                              '%s/blur_samples' % opt.outf + '.png')
            vutils.save_image(real_sim.mul(0.5).add(0.5),
                              '%s/sharp_samples' % opt.outf + '.png')

            fixed_blur = real_bim
            flag -= 1

        fake = netG(Variable(real_bim))

        contentLoss = criterion_L2(fake, Variable(real_sim))
        contentLoss.backward()
        errG = contentLoss

        optimizerG.step()

        ############################
        # (3) Report & 100 Batch checkpoint
        ############################
        if flag2:
            G1 = viz.line(
                np.array([errG.data[0]]), np.array([gen_iterations]),
                opts=dict(title='MSE loss toward real', caption='Gnet content loss')
            )
            flag2 -= 1
        else:
            viz.line(np.array([errG.data[0]]), np.array([gen_iterations]), update='append', win=G1)

        print('[%d/%d][%d/%d][%d] err_G: %f'
              % (epoch, opt.niter, iter_count, len(dataloader), gen_iterations, errG.data[0]))

        if gen_iterations % 100 == 0:
            fake = netG(Variable(fixed_blur, volatile=True))

            if flag3:
                imageW = viz.images(
                    fake.data.mul(0.5).add(0.5).cpu().clamp(0, 1).numpy(),
                    opts=dict(title='deblur img', caption='level final')
                )
                flag3 -= 1
            else:
                viz.images(
                    fake.data.mul(0.5).add(0.5).cpu().clamp(0, 1).numpy(),
                    win=imageW,
                    opts=dict(title='deblur img', caption='level final')
                )

        if gen_iterations % 1000 == 0:
            vutils.save_image(fake.data.mul(0.5).add(0.5),
                              '%s/fake_samples_gen_iter_%08d.png' % (opt.outf, gen_iterations))

        gen_iterations += 1

    # do checkpointing
    if epoch % opt.cut == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
