import torch
import torch.nn as nn
import torch.nn.functional as F
from mini_ml.utils.register import LOSSES


@LOSSES.register()
class PixelContrastCrossEntropyLoss(nn.Module):
    """
    Pixel-level contrastive loss for semantic segmentation with temperature scaling.

    Args:
        temperature (float): 温度系数。
        base_temperature (float): 基础温度，用于缩放。
        ignore_index (int): 标签中忽略的索引。
        max_samples (int): 每个类别最多采样的像素数。
        max_views (int): 每个 anchor 的正样本最大视角数。
    """

    def __init__(
        self,
        temperature=0.1,
        base_temperature=0.07,
        ignore_index=255,
        max_samples=1024,
        max_views=100,
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.ignore_index = ignore_index
        self.max_samples = max_samples
        self.max_views = max_views

    def forward(self, embeddings, labels):
        """
        embeddings: [B, C, H, W]
        labels: [B, H, W]
        """
        B, C, H, W = embeddings.shape
        embeddings = embeddings.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        labels = labels.view(-1)  # [B*H*W]

        mask = labels != self.ignore_index
        embeddings = embeddings[mask]
        labels = labels[mask]

        if len(labels) == 0:
            return torch.tensor(0.0, device=embeddings.device)

        embeddings = F.normalize(embeddings, dim=1)

        # 对每个类别采样固定数量像素
        unique_labels = labels.unique()
        sampled_idx = []
        for lbl in unique_labels:
            idx = (labels == lbl).nonzero(as_tuple=True)[0]
            if len(idx) > self.max_samples:
                idx = idx[torch.randperm(len(idx))[: self.max_samples]]
            sampled_idx.append(idx)
        if len(sampled_idx) > 0:
            sampled_idx = torch.cat(sampled_idx, dim=0)
        else:
            sampled_idx = torch.arange(len(labels), device=labels.device)

        embeddings = embeddings[sampled_idx]
        labels = labels[sampled_idx]

        # 计算相似度矩阵，并用 base_temperature 缩放
        logits = torch.matmul(embeddings, embeddings.T) / self.temperature
        logits = logits * (self.base_temperature / self.temperature)

        labels_eq = labels.unsqueeze(1) == labels.unsqueeze(0)  # [N, N]

        # 去掉自身对比
        logits = logits - torch.eye(len(logits), device=logits.device) * 1e9

        # max_views: 对每个 anchor 只保留 max_views 个正样本
        mask_pos = labels_eq.clone()
        N = labels.shape[0]
        for i in range(N):
            pos_idx = torch.where(labels_eq[i])[0]
            if len(pos_idx) > self.max_views:
                selected = pos_idx[torch.randperm(len(pos_idx))[: self.max_views]]
                mask_pos[i] = False
                mask_pos[i, selected] = True

        positives = logits[mask_pos]
        loss = -torch.log(torch.exp(positives).sum(0) / torch.exp(logits).sum(0)).mean()
        return loss
