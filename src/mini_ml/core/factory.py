from typing import Union

from pydantic import BaseModel
from torch import nn, optim
from torch.utils.data import Dataset

from mini_ml.dataset.trans import Compose
from mini_ml.models.losses.combined_loss import CombinedLoss
from mini_ml.utils.config import (
    CombinedLossConfig,
    ModelConfig,
    OptimizerConfig,
    LR_SchedulerConfig,
)

# 假设你已经有了所有组件的注册表
from mini_ml.utils.register import (
    ARCH,
    BACKBONES,
    DATASETS,
    HEADERS,
    LOSSES,
    OPTIMIZERS,
    TRANSFORMS,
    SCHEDULERS,
)


def build_from_cfg(cfg: Union[BaseModel, dict], registry: dict, **kwargs):
    """
    一个通用的、支持嵌套构建的组件工厂。
    这是整个框架的“引擎”。
    """
    if isinstance(cfg, BaseModel):
        cfg_dict = cfg.model_dump()
    else:
        cfg_dict = cfg.copy()

    component_name = cfg_dict.pop("type")
    builder = registry.get(component_name)
    if builder is None:
        raise ValueError(f"Component '{component_name}' not found in {registry.name}.")

    # --- 核心魔法：递归构建嵌套组件 ---
    for key, value in cfg_dict.items():
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], dict)
            and "type" in value[0]
        ):
            # 案例: 'losses' 参数是一个组件列表
            # 递归地为这个列表中的每一项构建组件
            # 注意: 这里我们假设所有子组件都在 LOSSES 注册表中，可以根据需要扩展
            cfg_dict[key] = [build_from_cfg(sub_cfg, LOSSES) for sub_cfg in value]
        elif isinstance(value, dict) and "type" in value:
            # 案例: 'backbone' 参数是一个单一组件
            # 递归地构建这个子组件 (例如，从 BACKBONES 注册表)
            # 这个逻辑在 build_model 中被更明确地处理了
            pass  # 在这里我们简化，只处理列表情况

    cfg_dict.update(kwargs)
    return builder(**cfg_dict)


# ---------------------------------------------------------------------------
# 2. 所有具体的构建函数都变成了简单的“一行”包装
# ---------------------------------------------------------------------------


def build_model(cfg: ModelConfig) -> nn.Module:
    """构建完整的模型。"""
    # 这里的逻辑更清晰：先构建零件，再组装
    backbone = build_from_cfg(cfg.backbone, BACKBONES)
    head = build_from_cfg(cfg.head, HEADERS, in_channels=backbone.out_channels)
    # 架构本身也是一个组件
    return build_from_cfg(cfg, ARCH, backbone=backbone, head=head)


def build_loss(cfg: CombinedLossConfig) -> nn.Module:
    """
    构建损失函数。现在它只是对通用工厂的一个简单调用。
    """
    sub_losses = [build_from_cfg(loss_cfg, LOSSES) for loss_cfg in cfg.types]
    return CombinedLoss(losses=sub_losses, coefficients=cfg.coef)


def build_optimizer(cfg: OptimizerConfig, model: nn.Module) -> optim.Optimizer:
    """构建优化器。"""
    return build_from_cfg(cfg, OPTIMIZERS, params=model.parameters())


def build_lr_scheduler(cfg: LR_SchedulerConfig, optimizer: optim.Optimizer, **kwargs):
    """构建学习率调度器。"""
    return build_from_cfg(cfg, SCHEDULERS, optimizer=optimizer, **kwargs)


def build_transforms(cfg_list: list) -> Compose:
    """根据配置列表构建一个数据变换流水线。"""
    transforms = []
    for transform_cfg in cfg_list:
        cfg = transform_cfg.copy()
        transform_name = cfg.pop("type")
        builder = TRANSFORMS.get(transform_name)
        if builder is None:
            raise ValueError(f"Transform '{transform_name}' is not registered.")
        transforms.append(builder(**cfg))
    return Compose(transforms)


def build_dataset(cfg: dict, mode: str) -> Dataset:
    """根据配置构建数据集。"""
    dataset_cfg = cfg[mode].copy()
    dataset_name = dataset_cfg.pop("type")

    # 1. 构建该数据集所需的数据变换
    transforms_pipeline = build_transforms(dataset_cfg.pop("transforms", []))

    # 2. 构建数据集实例
    builder = DATASETS.get(dataset_name)
    if builder is None:
        raise ValueError(f"Dataset '{dataset_name}' is not registered.")

    dataset_cfg["mode"] = mode
    dataset_cfg["transforms"] = transforms_pipeline

    return builder(**dataset_cfg)
