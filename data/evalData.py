from __future__ import division

import os
import os.path

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(root):
    images = []
    for fname in sorted(os.listdir(root)):
        if is_image_file(fname):
            path = os.path.join(root, fname)
            images.append(path)
    return images


def loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder_test(data.Dataset):
    def __init__(self, root, transform=None):  # , option=None):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in folders."))

        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        path = self.imgs[index]
        img = loader(path)
        return self.transform(img), os.path.split(path)[1]

    def __len__(self):
        return len(self.imgs)


def CreateDataLoader(opt):
    Trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_test = ImageFolder_test(root=opt.dataroot, transform=Trans)

    assert dataset_test

    return data.DataLoader(dataset_test, batch_size=1, num_workers=1, drop_last=False)
