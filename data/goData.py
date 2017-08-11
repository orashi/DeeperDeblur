from __future__ import division

import math
import os
import os.path
import random

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
from torchvision.transforms import Scale, CenterCrop

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


class RandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img1, img2):
        for attempt in range(10):
            area = img1.size[0] * img1.size[1]
            target_area = random.uniform(0.068, 0.08) * area  # fix this
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img1.size[0] and h <= img1.size[1]:
                x1 = random.randint(0, img1.size[0] - w)
                y1 = random.randint(0, img1.size[1] - h)

                img1 = img1.crop((x1, y1, x1 + w, y1 + h))
                img2 = img2.crop((x1, y1, x1 + w, y1 + h))
                assert (img1.size == (w, h))

                return img1.resize((self.size, self.size), self.interpolation), img2.resize((self.size, self.size),
                                                                                            self.interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img1)), crop(scale(img2))


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    for bag in sorted(os.listdir(dir)):
        root = os.path.join(dir, bag)
        for fname in sorted(os.listdir(os.path.join(root, 'blur_gamma'))):
            if is_image_file(fname):
                Bpath, Spath = os.path.join(root, 'blur_gamma', fname), os.path.join(root, 'sharp', fname)
                images.append((Bpath, Spath))
    return images


# TODO: abandon gamma

def loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None):  # , option=None):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in folders."))

        self.imgs = imgs
        self.transform = transform
        # self.opt = option

    def __getitem__(self, index):
        Bpath, Spath = self.imgs[index]
        Bimg, Simg = loader(Bpath), loader(Spath)

        #############################################
        Bimg, Simg = RandomSizedCrop(256, Image.BICUBIC)(Bimg, Simg)  # corp

        if random.random() < 0.5:  # rotate
            Bimg, Simg = Bimg.transpose(Image.FLIP_LEFT_RIGHT), Simg.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            Bimg, Simg = Bimg.transpose(Image.FLIP_TOP_BOTTOM), Simg.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            Bimg, Simg = Bimg.transpose(Image.ROTATE_90), Simg.transpose(Image.ROTATE_90)

        satRatio = random.uniform(0.5, 1.5)  # saturation
        Bimg, Simg = ImageEnhance.Color(Bimg).enhance(satRatio), ImageEnhance.Color(Simg).enhance(satRatio)

        # TODO: random noise & beyond 256 support
        #############################################

        result = []
        for i in reversed(range(3)):
            ratio = 256 // (2 ** i)
            result.append(self.transform(Bimg.resize((ratio, ratio), Image.BICUBIC)))
        for i in reversed(range(3)):
            ratio = 256 // (2 ** i)
            result.append(self.transform(Simg.resize((ratio, ratio), Image.BICUBIC)))

        return result

    def __len__(self):
        return len(self.imgs)


def CreateDataLoader(opt):
    random.seed(opt.manualSeed)

    Trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder(root=opt.dataroot, transform=Trans, option=opt)

    assert dataset

    return data.DataLoader(dataset, batch_size=opt.batchSize,
                           shuffle=True, num_workers=int(opt.workers), drop_last=True)
