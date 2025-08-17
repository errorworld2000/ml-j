import torch
from torch import nn

from mini_ml.utils.register import LOSSES


@LOSSES.register()
class CombinedLoss(nn.Module):
    """
    一个复合损失函数，用于计算多个损失函数的加权和。
    """

    def __init__(self, losses: list[nn.Module], coefficients: list[float]):
        """
        初始化复合损失。
        :param losses: 一个包含多个损失函数实例的列表 (e.g., [nn.CrossEntropyLoss(), DiceLoss()])。
        :param coefficients: 一个包含对应权重的浮点数列表 (e.g., [1.0, 0.5])。
        """
        super().__init__()

        if len(losses) != len(coefficients):
            raise ValueError(
                f"损失函数的数量 ({len(losses)}) 必须与系数的数量 ({len(coefficients)}) 相匹配。"
            )

        # 使用 nn.ModuleList 来确保 PyTorch 能正确地发现和管理这些子损失模块
        self.losses = nn.ModuleList(losses)
        self.coefficients = coefficients

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        计算加权总损失。
        :param predictions: 模型的预测输出。
        :param targets: 真实的标签。
        :return: 一个包含详细信息的字典:
                 {
                   'total_loss': torch.Tensor,  # 用于反向传播的总损失
                   'CrossEntropyLoss_0': float, # 第一个损失的标量值 (用于日志记录)
                   'DiceLoss_1': float,         # 第二个损失的标量值 (用于日志记录)
                   ...
                 }
        """
        loss_dict = {}
        total_loss = 0.0

        for i, (loss_fn, coef) in enumerate(zip(self.losses, self.coefficients)):
            # 获取损失函数的类名作为日志记录的键
            loss_name = loss_fn.__class__.__name__
            unique_loss_key = f"{loss_name}_{i}"

            # 计算单个损失 (这是一个张量)
            single_loss = loss_fn(predictions, targets)

            # 将加权的损失累加到总损失中
            total_loss += coef * single_loss

            # 将这个损失的标量值存入字典，用于日志记录和监控
            loss_dict[unique_loss_key] = single_loss.item()

        # 将最终的、可用于反向传播的总损失也放入字典中
        loss_dict["total_loss"] = total_loss

        return loss_dict
