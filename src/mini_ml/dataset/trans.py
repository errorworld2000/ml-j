import random

import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms

from mini_ml.utils.register import TRANSFORMS


@TRANSFORMS.register()
class Normalize:
    """一个包装器，用于 torchvision 的 Normalize。"""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = transforms.Normalize(mean=mean, std=std)

    def __call__(self, image, mask):
        return self.transform(image), mask


@TRANSFORMS.register()
class RandomHorizontalFlip:
    """对图像和掩码同时进行随机水平翻转。"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            return F.hflip(image), F.hflip(mask)
        return image, mask


@TRANSFORMS.register()
class Resize:
    """对图像和掩码进行调整大小。"""

    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        return F.resize(image, self.size), F.resize(
            mask, self.size, interpolation=Image.Resampling.NEAREST
        )


@TRANSFORMS.register()
class Compose:
    """将多个变换组合在一起。"""

    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
