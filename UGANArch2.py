import argparse
import os
import random
from functools import reduce
from math import log10

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.utils as vutils
from visdom import Visdom

from data.UData import CreateDataLoader
from models.UGANModel import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--testBatch', type=int, default=10, help='input test batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--cut', type=int, default=2, help='cut backup frequency')
parser.add_argument('--niter', type=int, default=700, help='number of epochs to train for')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--Diters', type=int, default=1, help='number of D iters per each G iter')
parser.add_argument('--manualSeed', type=int, default=2345, help='random seed to use. Default=1234')
parser.add_argument('--baseGeni', type=int, default=0, help='start base of pure pair L1 loss')
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
# random seed setup
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
cudnn.benchmark = True
####### regular set up end


viz = Visdom(env=opt.env)

dataloader_train, dataloader_test = CreateDataLoader(opt)

netG = Pyramid()
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = PatchD()
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion_GAN = GANLoss()
if opt.cuda:
    criterion_GAN = GANLoss(tensor=torch.cuda.FloatTensor)
criterion_L1 = nn.L1Loss()
criterion_L2 = nn.MSELoss()

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion_L1.cuda()
    criterion_L2.cuda()
    criterion_GAN.cuda()

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.9))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.9))
schedulerG = lr_scheduler.ReduceLROnPlateau(optimizerG, mode='max', verbose=True, min_lr=0.000005,
                                            patience=10)  # 1.5*10^5 iter
schedulerD = lr_scheduler.ReduceLROnPlateau(optimizerD, mode='max', verbose=True, min_lr=0.000005,
                                            patience=10)  # 1.5*10^5 iter

flag = 1
flag2 = 1
flag3 = 1
flag4 = 1
flag5 = 1
flag6 = 1
for epoch in range(opt.epoi, opt.niter):

    epoch_loss = 0
    epoch_iter_count = 0

    for extra in range(4):
        data_iter = iter(dataloader_train)
        iter_count = 0

        while iter_count < len(dataloader_train):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            for p in netG.parameters():
                p.requires_grad = False  # to avoid computation

            # train the discriminator Diters times
            Diters = opt.Diters

            if gen_iterations < opt.baseGeni:  # L1 stage
                Diters = 0

            j = 0
            while j < Diters and iter_count < len(dataloader_train):

                j += 1
                netD.zero_grad()

                data = data_iter.next()
                iter_count += 1

                if opt.cuda:
                    data = [x.cuda() for x in data]
                real_bim, real_sim = data[0:3], data[3:]

                # train with fake

                fake_Vsim = netG(Variable(real_bim[2], volatile=True))

                errD_fake = reduce(lambda x, y: 0.5 * x + y,
                                   map(lambda x, y: criterion_GAN(netD(Variable(torch.cat([x.data, y], 1))),
                                                                  False), fake_Vsim, real_bim))
                errD_fake.backward(retain_graph=True)  # backward on score on real

                errD_real = reduce(lambda x, y: 0.5 * x + y,
                                   map(lambda x, y: criterion_GAN(netD(Variable(torch.cat([x, y], 1))),
                                                                  True), real_sim, real_bim))
                errD_real.backward()  # backward on score on real

                errD = errD_real + errD_fake

                optimizerD.step()
            ############################
            # (2) Update G network
            ############################
            if iter_count < len(dataloader_train):

                for p in netD.parameters():
                    p.requires_grad = False  # to avoid computation
                for p in netG.parameters():
                    p.requires_grad = True  # to avoid computation
                netG.zero_grad()

                data = data_iter.next()
                iter_count += 1

                if opt.cuda:
                    data = [x.cuda() for x in data]

                real_bim, real_sim = data[0:3], data[3:]

                if flag:  # fix samples
                    for i in range(3):
                        viz.images(
                            real_bim[i].mul(0.5).add(0.5).cpu().numpy(),
                            opts=dict(title='blur img', caption='level ' + str(i + 1))
                        )
                        viz.images(
                            real_sim[i].mul(0.5).add(0.5).cpu().numpy(),
                            opts=dict(title='sharp img', caption='level ' + str(i + 1))
                        )

                        vutils.save_image(real_bim[i].mul(0.5).add(0.5),
                                          '%s/blur_samples' % opt.outf + str(i + 1) + '.png')
                        vutils.save_image(real_sim[i].mul(0.5).add(0.5),
                                          '%s/sharp_samples' % opt.outf + str(i + 1) + '.png')

                    fixed_blur = real_bim[2]
                    flag -= 1

                fake = netG(Variable(real_bim[2]))

                if gen_iterations < opt.baseGeni:
                    contentLoss = reduce(lambda x, y: x + y,
                                         map(lambda x, y: criterion_L2(x.mul(0.5).add(0.5), Variable(y.mul(0.5).add(0.5))), fake, real_sim)) / 3
                    contentLoss.backward()
                    epoch_loss += 10 * log10(1 / contentLoss.data[0])
                    epoch_iter_count += 1
                    errG = contentLoss
                else:
                    errG = reduce(lambda x, y: 0.5 * x + y,
                                  map(lambda x, y: criterion_GAN(netD(torch.cat([x, Variable(y)], 1)), True) * 0.0001,
                                      fake,
                                      real_bim))
                    errG.backward(retain_graph=True)

                    contentLoss = reduce(lambda x, y: x + y,
                                         map(lambda x, y: criterion_L2(x.mul(0.5).add(0.5), Variable(y.mul(0.5).add(0.5))), fake, real_sim)) / 3
                    contentLoss.backward()

                    epoch_loss += 10 * log10(1 / contentLoss.data[0])
                    epoch_iter_count += 1

                optimizerG.step()

            ############################
            # (3) Report & 100 Batch checkpoint
            ############################

            if gen_iterations < opt.baseGeni:
                if flag2:
                    L1window = viz.line(
                        np.array([contentLoss.data[0]]), np.array([gen_iterations]),
                        opts=dict(title='MSE loss toward real', caption='Gnet content loss')
                    )
                    flag2 -= 1
                else:
                    viz.line(np.array([contentLoss.data[0]]), np.array([gen_iterations]), update='append', win=L1window)

                print('[%d/%d][%d/%d][%d] err_G: %f'
                      % (epoch, opt.niter, iter_count + extra * len(dataloader_train), len(dataloader_train) * 4,
                         gen_iterations, contentLoss.data[0]))
            else:
                if flag4:
                    D1 = viz.line(
                        np.array([errD.data[0]]), np.array([gen_iterations]),
                        opts=dict(title='errD(distinguishability)', caption='total Dloss')
                    )
                    D2 = viz.line(
                        np.array([errD_real.data[0]]), np.array([gen_iterations]),
                        opts=dict(title='errD_real', caption='real\'s mistake')
                    )
                    D3 = viz.line(
                        np.array([errD_fake.data[0]]), np.array([gen_iterations]),
                        opts=dict(title='errD_fake', caption='fake\'s mistake')
                    )
                    G1 = viz.line(
                        np.array([errG.data[0]]), np.array([gen_iterations]),
                        opts=dict(title='Gnet loss toward real', caption='Gnet loss')
                    )
                    flag4 -= 1
                if flag2:
                    L1window = viz.line(
                        np.array([contentLoss.data[0]]), np.array([gen_iterations]),
                        opts=dict(title='MSE loss toward real', caption='Gnet content loss')
                    )
                    flag2 -= 1

                viz.line(np.array([errD.data[0]]), np.array([gen_iterations]), update='append', win=D1)
                viz.line(np.array([errD_real.data[0]]), np.array([gen_iterations]), update='append', win=D2)
                viz.line(np.array([errD_fake.data[0]]), np.array([gen_iterations]), update='append', win=D3)
                viz.line(np.array([errG.data[0]]), np.array([gen_iterations]), update='append', win=G1)
                viz.line(np.array([contentLoss.data[0]]), np.array([gen_iterations]), update='append', win=L1window)

                print('[%d/%d][%d/%d][%d] errD: %f err_G: %f err_D_real: %f err_D_fake %f content loss %f'
                      % (epoch, opt.niter, iter_count + extra * len(dataloader_train), len(dataloader_train) * 4,
                         gen_iterations,
                         errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0], contentLoss.data[0]))

            if gen_iterations % 100 == 0:
                fake = netG(Variable(fixed_blur, volatile=True))

                if flag3:
                    imageW = []
                    for i in range(3):
                        imageW.append(viz.images(
                            fake[i].data.mul(0.5).add(0.5).clamp(0, 1).cpu().numpy(),
                            opts=dict(title='deblur img', caption='level ' + str(i + 1))
                        ))
                    flag3 -= 1
                else:
                    for i in range(3):
                        viz.images(
                            fake[i].data.mul(0.5).add(0.5).clamp(0, 1).cpu().numpy(),
                            win=imageW[i],
                            opts=dict(title='deblur img', caption='level ' + str(i + 1))
                        )

            if gen_iterations % 1000 == 0:
                vutils.save_image(fake[2].data.mul(0.5).add(0.5),
                                  '%s/fake_samples_gen_iter_%08d.png' % (opt.outf, gen_iterations))

            gen_iterations += 1

    # if epoch % 5 == 0:
    #     avg_psnr = 0
    #     for batch in dataloader_test:
    #         input, target = [x.cuda() for x in batch]
    #         prediction = netG(Variable(input, volatile=True))
    #         mse = criterion_L2(prediction[2].mul(0.5).add(0.5), Variable(target.mul(0.5).add(0.5)))
    #         psnr = 10 * log10(1 / mse.data[0])
    #         avg_psnr += psnr
    #     avg_psnr = avg_psnr / len(dataloader_test)

    #     if flag6:
    #         Test = viz.line(
    #             np.array([avg_psnr]), np.array([epoch]),
    #             opts=dict(title='Test PSNR', caption='PSNR')
    #         )
    #         flag6 -= 1
    #     else:
    #         viz.line(np.array([avg_psnr]), np.array([epoch]), update='append', win=Test)
    #     print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))


    if flag5:
        epoL = viz.line(
            np.array([epoch_loss / epoch_iter_count]), np.array([epoch]),
            opts=dict(title='Train epoch PSNR', caption='Epoch PSNR')
        )
        flag5 -= 1
        schedulerG.step(epoch_loss / epoch_iter_count)
        schedulerD.step(epoch_loss / epoch_iter_count)
    else:
        viz.line(np.array([epoch_loss / epoch_iter_count]), np.array([epoch]), update='append', win=epoL)
        schedulerG.step(epoch_loss / epoch_iter_count)
        schedulerD.step(epoch_loss / epoch_iter_count)

    # do checkpointing
    if opt.cut == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_only.pth' % opt.outf)
        torch.save(netD.state_dict(), '%s/netD_epoch_only.pth' % opt.outf)
    elif epoch % opt.cut == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

        # TODO: max logD?
