import pytest
import torch
from src.rewrite_model.models.hrnet import hrnet18


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
        model = hrnet18(pretrained=False)
        assert model is not None
        output=model(torch.randn(1, 3, 256, 256))

