from torch import nn
import torch
import torch.nn.functional as F
from mini_ml.utils.register import ARCHS


@ARCHS.register()
class HRNetSegmentor(nn.Module):
    """HRNet 分割器模型实现。"""

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        drop_prob: float = -1,
        proj_dim: int = -1,
    ):

        super().__init__()
        self.backbone = backbone
        self.head = head
        if drop_prob and drop_prob > 0:
            self.dropout = nn.Dropout(p=drop_prob)
        else:
            self.dropout = None

        if proj_dim > 0:
            last_channels = getattr(backbone, "out_channels", None)
            if last_channels is None:
                raise ValueError("backbone 必须定义 out_channels 用于 ProjectionHead")
            self.proj_head = ProjectionHead(dim_in=last_channels, proj_dim=proj_dim)
        else:
            self.proj_head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if self.dropout is not None:
            features = [self.dropout(f) for f in features]
        predictions = self.head(features)
        outputs = {"pred": predictions}
        if self.training and self.proj_head is not None:
            feats = features[-1]
            outputs["embed"] = self.proj_head(feats)

        return outputs


class ProjectionHead(nn.Module):
    """
    投影层，用于对比学习
    """

    def __init__(self, dim_in, proj_dim=256, proj="convmlp"):
        super().__init__()
        if proj == "linear":
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == "convmlp":
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1),
            )
        else:
            raise ValueError(f"Unsupported proj type: {proj}")

    def forward(self, x):
        # L2 normalize embeddings
        return F.normalize(self.proj(x), p=2, dim=1)
