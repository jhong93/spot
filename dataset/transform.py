import random
import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class RandomHorizontalFlipFLow(nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if torch.rand(1)[0] < self.p:
            shape = img.shape
            img.view((-1,) + shape[-3:])[:, 1, :, :] *= -1
            return img.flip(-1)
        return img


class RandomOffsetFlow(nn.Module):

    def __init__(self, p=0.5, x=0.1, y=0.05):
        super().__init__()
        self.p = p
        self.x = x
        self.y = y

    def forward(self, img):
        if torch.rand(1)[0] < self.p:
            shape = img.shape
            view = img.view((-1,) + shape[-3:])
            view[:, 1, :, :] += (
                torch.rand(1, device=img.device)[0] * 2 - 1) * self.x
            view[:, 0, :, :] += (
                torch.rand(1, device=img.device)[0] * 2 - 1) * self.y
        return img


class RandomGaussianNoise(nn.Module):

    def __init__(self, p=0.5, s=0.1):
        super().__init__()
        self.p = p
        self.std = s ** 0.5

    def forward(self, img):
        v = torch.rand(1)[0]
        if v < self.p:
            img += torch.randn(img.shape, device=img.device) * self.std
        return img


class SeedableRandomSquareCrop:

    def __init__(self, dim):
        self._dim = dim

    def __call__(self, img):
        c, h, w = img.shape[-3:]
        x, y = 0, 0
        if h > self._dim:
            y = random.randint(0, h - self._dim)
        if w > self._dim:
            x = random.randint(0, w - self._dim)
        return F.crop(img, y, x, self._dim, self._dim)


class ThreeCrop:

    def __init__(self, dim):
        self._dim = dim

    def __call__(self, img):
        c, h, w = img.shape[-3:]
        y = (h - self._dim) // 2
        ret = []
        dw = w - self._dim
        for x in (0, dw // 2, dw):
            ret.append(F.crop(img, y, x, self._dim, self._dim))
        return torch.stack(ret)