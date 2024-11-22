
import pytest
import torch
from rewrite_model.models.backbones.hrnet import HRNet_W32


@pytest.mark.pytorch
class TestHRNet:
    def test_create_model(self):
        """
        测试创建模型的函数

        Args:
            无

        Returns:
            无

        Raises:
            AssertionError: 如果模型为None，则抛出断言错误

        """
        model = HRNet_W32(pretrained="https://huggingface.co/timm/hrnet_w18.ms_aug_in1k/blob/main/model.safetensors")
        assert model is not None
        data = torch.randint(0, 256, (1, 3, 256, 196), dtype=torch.float32)
        y = model(data)
        for idx, i in enumerate(y):
            print(f"output shape of layer{idx}: {i.shape}")
