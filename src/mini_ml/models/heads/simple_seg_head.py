from torch import nn
from mini_ml.utils.register import HEADS


@HEADS.register()
class SimpleSegmentationHead(nn.Module):
    """简单的分割头部实现。"""

    def __init__(self, in_channels: int, num_classes: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, features: list) -> nn.Module:
        # 主干网络的输出是一个列表，我们取第一个元素
        x = features[0]
        return self.conv(x)
