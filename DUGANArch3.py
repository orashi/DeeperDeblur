import argparse
import os
import random
from math import log10

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable, grad

from data.DUGData import CreateDataLoader
from models.CascadeNeXt import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--test', action='store_true', help='test option')
parser.add_argument('--adv', action='store_true', help='adversarial training option')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--testBatch', type=int, default=4, help='input test batch size')
parser.add_argument('--cut', type=int, default=2, help='cut backup frequency')
parser.add_argument('--niter', type=int, default=700, help='number of epochs to train for')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--advW', type=float, default=0.0001, help='adversarial weight, default=0.0001')
parser.add_argument('--gpW', type=float, default=10, help='gradient penalty weight')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.9')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--optim', action='store_true', help='load optimizer\'s checkpoint')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--Diters', type=int, default=1, help='number of D iters per each G iter')
parser.add_argument('--manualSeed', type=int, default=2345, help='random seed to use. Default=1234')
parser.add_argument('--baseGeni', type=int, default=0, help='start base of pure pair L1 loss')
parser.add_argument('--geni', type=int, default=0, help='continue gen image num')
parser.add_argument('--epoi', type=int, default=0, help='continue epoch num')
parser.add_argument('--env', type=str, default=None, help='tensorboard env')

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

writer = SummaryWriter(log_dir=opt.env, comment='this is great')

dataloader_train, dataloader_test = CreateDataLoader(opt)

netG = Pyramid()
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = PatchD()
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()
criterion_L2 = nn.MSELoss()
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion_L1.cuda()
    criterion_L2.cuda()
    criterion_GAN.cuda()
    one, mone = one.cuda(), mone.cuda()

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.9))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.9))
if opt.optim:
    optimizerG.load_state_dict(torch.load('%s/optimG_checkpoint.pth' % opt.outf))
    optimizerD.load_state_dict(torch.load('%s/optimD_checkpoint.pth' % opt.outf))

schedulerG = lr_scheduler.ReduceLROnPlateau(optimizerG, mode='max', verbose=True, min_lr=0.0000005,
                                            patience=10)  # 1.5*10^5 iter
schedulerD = lr_scheduler.ReduceLROnPlateau(optimizerD, mode='max', verbose=True, min_lr=0.0000005,
                                            patience=10)  # 1.5*10^5 iter


def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(opt.batchSize, 1, 1, 1)
    # alpha = alpha.expand(opt.batchSize, real_data.nelement() / opt.batchSize).contiguous().view(opt.batchSize, 3, 64,
    #                                                                                             64)
    alpha = alpha.cuda() if opt.cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                     grad_outputs=torch.ones(disc_interpolates.size()).cuda() if opt.cuda else torch.ones(
                         disc_interpolates.size()),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.gpW
    return gradient_penalty


flag = 1
for epoch in range(opt.epoi, opt.niter):

    epoch_loss = 0
    epoch_iter_count = 0

    for extra in range(2 * (opt.Diters + 1)):
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

            if gen_iterations < opt.baseGeni or not opt.adv:  # L1 stage
                Diters = 0

            j = 0
            while j < Diters and iter_count < len(dataloader_train):

                j += 1
                netD.zero_grad()

                real_bim, real_sim = data_iter.next()
                iter_count += 1

                if opt.cuda:
                    real_bim, real_sim = real_bim.cuda(), real_sim.cuda()

                # train with fake

                fake_sim = netG(Variable(real_bim, volatile=True)).data

                errD_fake = netD(Variable(torch.cat([fake_sim, real_bim], 1))).mean(0).view(1)
                errD_fake.backward(one, retain_graph=True)  # backward on score on real

                errD_real = netD(Variable(torch.cat([real_sim, real_bim], 1))).mean(0).view(1)
                errD_real.backward(mone, retain_graph=True)  # backward on score on real

                errD = errD_real - errD_fake

                # gradient penalty
                gradient_penalty = calc_gradient_penalty(netD, torch.cat([real_sim, real_bim], 1),
                                                         torch.cat([fake_sim, real_bim], 1))
                gradient_penalty.backward()

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

                real_bim, real_sim = data_iter.next()
                iter_count += 1

                if opt.cuda:
                    real_bim, real_sim = real_bim.cuda(), real_sim.cuda()

                if flag:  # fix samples
                    writer.add_image('target imgs', vutils.make_grid(real_sim.mul(0.5).add(0.5), nrow=16))
                    writer.add_image('blur imgs', vutils.make_grid(real_bim.mul(0.5).add(0.5), nrow=16))
                    vutils.save_image(real_sim.mul(0.5).add(0.5),
                                      '%s/sharp_samples' % opt.outf + '.png')
                    vutils.save_image(real_bim.mul(0.5).add(0.5),
                                      '%s/blur_samples' % opt.outf + '.png')
                    fixed_blur = real_bim
                    flag -= 1

                fake = netG(Variable(real_bim))

                if gen_iterations < opt.baseGeni or not opt.adv:
                    contentLoss = criterion_L2(fake.mul(0.5).add(0.5), Variable(real_sim.mul(0.5).add(0.5)))
                    contentLoss.backward()

                    epoch_loss += 10 * log10(1 / contentLoss.data[0])
                    epoch_iter_count += 1
                    errG = contentLoss
                else:
                    errG = netD(torch.cat([fake, Variable(real_bim)], 1)).mean(0).view(1) * opt.advW
                    errG.backward(mone, retain_graph=True)

                    contentLoss = criterion_L2(fake.mul(0.5).add(0.5), Variable(real_sim.mul(0.5).add(0.5)))
                    contentLoss.backward()

                    epoch_loss += 10 * log10(1 / contentLoss.data[0])
                    epoch_iter_count += 1

                optimizerG.step()

            ############################
            # (3) Report & 100 Batch checkpoint
            ############################

            if gen_iterations < opt.baseGeni or not opt.adv:
                writer.add_scalar('MSE Loss', contentLoss.data[0], gen_iterations)
                print('[%d/%d][%d/%d][%d] err_G: %f'
                      % (epoch, opt.niter, iter_count + extra * len(dataloader_train),
                         len(dataloader_train) * 2 * (opt.Diters + 1), gen_iterations, contentLoss.data[0]))
            else:
                writer.add_scalar('MSE Loss', contentLoss.data[0], gen_iterations)
                writer.add_scalar('wasserstein distance', errD.data[0], gen_iterations)
                writer.add_scalar('errD_real', errD_real.data[0], gen_iterations)
                writer.add_scalar('errD_fake', errD_fake.data[0], gen_iterations)
                writer.add_scalar('Gnet loss toward real', errG.data[0], gen_iterations)
                writer.add_scalar('gradient_penalty', gradient_penalty.data[0], gen_iterations)
                print('[%d/%d][%d/%d][%d] errD: %f err_G: %f err_D_real: %f err_D_fake %f content loss %f'
                      % (epoch, opt.niter, iter_count + extra * len(dataloader_train), len(dataloader_train) * 2 * (opt.Diters + 1),
                         gen_iterations, errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0],
                         contentLoss.data[0]))

            if gen_iterations % 100 == 0:
                fake = netG(Variable(fixed_blur, volatile=True))
                writer.add_image('deblur imgs', vutils.make_grid(fake.data.mul(0.5).add(0.5).clamp(0, 1), nrow=16), gen_iterations)

            if gen_iterations % 1000 == 0:
                for name, param in netG.named_parameters():
                    writer.add_histogram('netG ' + name, param.clone().cpu().data.numpy(), gen_iterations)
                for name, param in netD.named_parameters():
                    writer.add_histogram('netD ' + name, param.clone().cpu().data.numpy(), gen_iterations)
                vutils.save_image(fake.data.mul(0.5).add(0.5),
                                  '%s/fake_samples_gen_iter_%08d.png' % (opt.outf, gen_iterations))

            gen_iterations += 1

    if opt.test:
        if epoch % 5 == 0:
            avg_psnr = 0
            for batch in dataloader_test:
                input, target = [x.cuda() for x in batch]
                prediction = netG(Variable(input, volatile=True))
                mse = criterion_L2(prediction.mul(0.5).add(0.5), Variable(target.mul(0.5).add(0.5)))
                psnr = 10 * log10(1 / mse.data[0])
                avg_psnr += psnr
            avg_psnr = avg_psnr / len(dataloader_test)

            writer.add_scalar('Test epoch PSNR', avg_psnr, epoch)

            print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))

    avg_psnr = epoch_loss / epoch_iter_count
    writer.add_scalar('Train epoch PSNR', avg_psnr, epoch)
    schedulerG.step(avg_psnr)
    schedulerD.step(avg_psnr)

    # do checkpointing
    if opt.cut == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_only.pth' % opt.outf)
        torch.save(netD.state_dict(), '%s/netD_epoch_only.pth' % opt.outf)
    elif epoch % opt.cut == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(optimizerG.state_dict(), '%s/optimG_checkpoint.pth' % opt.outf)
    torch.save(optimizerD.state_dict(), '%s/optimD_checkpoint.pth' % opt.outf)
