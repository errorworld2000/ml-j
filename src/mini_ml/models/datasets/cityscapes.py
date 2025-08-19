import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from mini_ml.utils.register import DATASETS


@DATASETS.register()
class CityscapesDataset(Dataset):
    def __init__(self, root_path: str, mode: str = "train", transforms=None):
        super().__init__()
        self.root_path = Path(root_path)
        self.transforms = transforms
        self.mode = mode

        self.image_paths = []
        self.mask_paths = []

        # 扫描文件
        img_dir = self.root_path / "leftImg8bit" / self.mode
        mask_dir = self.root_path / "gtFine" / self.mode

        for city_folder in sorted(os.listdir(img_dir)):
            city_img_dir = img_dir / city_folder
            city_mask_dir = mask_dir / city_folder
            for file_name in sorted(os.listdir(city_img_dir)):
                if file_name.endswith(".png"):
                    self.image_paths.append(city_img_dir / file_name)
                    # 找到对应的 mask 文件
                    mask_name = file_name.replace(
                        "_leftImg8bit.png", "_gtFine_labelIds.png"
                    )
                    self.mask_paths.append(city_mask_dir / mask_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像和掩码
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        # 应用数据变换
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        # 将 PIL Image 转换为 Tensor
        # 注意：这里的 toTensor 会将图像归一化到 [0, 1]
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image, mask
