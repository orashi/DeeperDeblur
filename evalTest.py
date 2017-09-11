import argparse

import torchvision.utils as vutils
from torch.autograd import Variable

from data.evalData import *
from models.CascadeNeXt import *

parser = argparse.ArgumentParser()
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
opt = parser.parse_args()
print(opt)

####### regular set up
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# cudnn.benchmark = True
####### regular set up end


dataloader_test = CreateDataLoader(opt)

netG = Pyramid().eval()
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

if opt.cuda:
    netG.cuda()
try:
    os.makedirs(os.path.join(os.path.split(opt.dataroot)[0], os.path.split(opt.dataroot)[1] + ' result'))
except OSError:
    pass
for input, name in dataloader_test:
    prediction = netG(Variable(input.cuda() if opt.cuda else input, volatile=True))
    vutils.save_image(torch.squeeze(prediction.data).mul(0.5).add(0.5),
                      os.path.join(os.path.split(opt.dataroot)[0], os.path.split(opt.dataroot)[1] + 'result', name[0]),
                      padding=0)
    print(name[0] + '　処理完了')
    prediction = 0
    input = 0
