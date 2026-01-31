import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms

from mini_ml.utils.register import TRANSFORMS


@TRANSFORMS.register()
class Normalize:
    """一个包装器，用于 torchvision 的 Normalize。"""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = transforms.Normalize(mean=mean, std=std)

    def __call__(self, image, mask=None):
        return self.transform(image), mask


@TRANSFORMS.register()
class RandomHorizontalFlip:
    """对图像和掩码同时进行随机水平翻转。"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask=None):
        if random.random() < self.p:
            image = F.hflip(image)
            if mask is not None:
                mask = F.hflip(mask)
        return image, mask


@TRANSFORMS.register()
class Resize:
    """对图像和掩码进行调整大小。"""

    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask=None):
        image = F.resize(
            image, self.size, interpolation=transforms.InterpolationMode.BILINEAR
        )
        if mask is not None:
            mask = F.resize(
                mask, self.size, interpolation=transforms.InterpolationMode.NEAREST
            )
        return image, mask


@TRANSFORMS.register()
class ToTensor:
    """把 PIL Image 或 np.ndarray 转换成 Tensor，并把 mask 转为 long。"""

    def __call__(self, image, mask=None):
        # PIL → Tensor
        if isinstance(image, Image.Image):
            image = F.to_tensor(image)
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if mask is not None:
            if isinstance(mask, Image.Image):
                mask = torch.from_numpy(np.array(mask)).long()
            elif isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).long()

        return image, mask
