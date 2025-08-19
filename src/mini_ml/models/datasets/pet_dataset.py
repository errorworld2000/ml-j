from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from mini_ml.utils.register import DATASETS


@DATASETS.register()
class PetDataset(Dataset):
    def __init__(self, root_path: str, mode: str = "train", transforms=None):
        super().__init__()
        self.root_path = Path(root_path)
        self.transforms = transforms

        image_dir = self.root_path / "images"
        mask_dir = self.root_path / "annotations" / "trimaps"

        # 1. 读取所有图像文件名 (不含后缀)
        #    文件名格式为: Breed_Number.jpg
        self.image_names = [f.stem for f in image_dir.glob("*.jpg")]

        # (可选) 划分训练集和验证集
        # 这里为了简单，我们用所有数据，但在真实项目中需要划分
        # random.shuffle(self.image_names)
        # num_train = int(len(self.image_names) * 0.8)
        # if mode == 'train':
        #     self.image_names = self.image_names[:num_train]
        # else:
        #     self.image_names = self.image_names[num_train:]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 2. 根据文件名构建完整的图像和掩码路径
        name = self.image_names[idx]
        image_path = self.root_path / "images" / f"{name}.jpg"
        mask_path = self.root_path / "annotations" / "trimaps" / f"{name}.png"

        # 3. 加载图像和掩码
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)  # 掩码是单通道的，无需 convert

        # 4. 应用数据变换
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        # 5. 将掩码的标签值从 (1, 2, 3) 转换为 (0, 1, 2)
        #    这是因为 CrossEntropyLoss 期望的标签是从 0 开始的
        mask = np.array(mask, dtype=np.int64) - 1
        mask = torch.from_numpy(mask)

        return image, mask
