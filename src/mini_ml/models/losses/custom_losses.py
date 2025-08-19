import torch
from torch import nn
from mini_ml.utils.register import LOSSES


@LOSSES.register()
class CrossEntropyLoss(nn.Module):
    """标准交叉熵损失包装，方便注册使用。"""

    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        return self.loss_fn(predictions, targets)
