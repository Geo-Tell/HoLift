import random
import torch
import torchvision.transforms.functional as F
from PIL import Image

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.shape[-2:] == lb.shape[-2:]
        H, W = self.size
        h, w = im.shape[-2:]
        if (H, W) == (h, w): return dict(im = im, lb = lb)
        if w < W or h < H:
            im_shape = list(im.shape)
            im_shape[-2:] = (H, W)
            lb_shape = list(lb.shape)
            lb_shape[-2:] = (H, W)
            image_padding = torch.zeros(im_shape)
            label_padding = torch.ones(lb_shape, dtype=torch.int64) * 255
            col_init = random.randint(0,W-w)
            row_init = random.randint(0,H-h)
            image_padding[..., row_init: row_init+h, col_init: col_init+w] = im
            label_padding[..., row_init: row_init+h, col_init: col_init+w] = lb
            im = image_padding
            lb = label_padding
            w = W
            h = H
        sw, sh = random.randint(0, w - W), random.randint(0, h - H)
        return dict(im = F.crop(im, int(sh), int(sw), H, W),
                    lb = F.crop(lb, int(sh), int(sw), H, W))

class RandomRotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        angle = int(random.choice(self.angle))
        return dict(im = F.rotate(im, angle),
                    lb = F.rotate(lb, angle, Image.NEAREST))

class HorizontalFlip(object):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']
            return dict(im=F.hflip(im), lb=F.hflip(lb))

class VerticalFlip(object):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']
            return dict(im=F.vflip(im), lb=F.vflip(lb))

class RandomScale(object):
    def __init__(self, scales = (1, )):
        self.scales = scales

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        H, W = im.shape[-2:]
        scale = random.choice(self.scales)
        h, w = int(H * scale + 0.5), int(W * scale + 0.5)
        return dict(im = F.resize(im, (h, w), Image.BILINEAR),
                    lb = F.resize(lb, (h, w), Image.NEAREST))

class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb