import argparse
from math import log10

import torch.backends.cudnn as cudnn

from data.oneData import CreateDataLoader
from models.oneModel import *

parser = argparse.ArgumentParser()
parser.add_argument('--testBatch', type=int, default=1, help='input test batch size')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")

opt = parser.parse_args()
print(opt)

####### regular set up
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
cudnn.benchmark = True
####### regular set up end


_, dataloader_test = CreateDataLoader(opt)

netG = Pyramid().eval()
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

criterion_L2 = nn.MSELoss()

if opt.cuda:
    netG.cuda()
    criterion_L2.cuda()


avg_psnr = 0
for batch in dataloader_test:
    input, target = [x.cuda() for x in batch]
    prediction = netG(Variable(input, volatile=True))
    mse = criterion_L2(prediction.mul(0.5).add(0.5), Variable(target.mul(0.5).add(0.5)))
    psnr = 10 * log10(1 / mse.data[0])
    print("PSNR: {:.4f} dB".format(psnr))

    avg_psnr += psnr
avg_psnr = avg_psnr / len(dataloader_test)


print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))

