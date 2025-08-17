from torch import nn
from mini_ml.utils.register import ARCH


@ARCH.register()
class HRNetSegmentor(nn.Module):
    """HRNet 分割器模型实现。"""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions
