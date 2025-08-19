from typing import Literal, Optional, Union

from pydantic import BaseModel


# --- 定义所有可配置的“组件”的基类 ---
class ComponentConfig(BaseModel):
    """所有组件配置的基类，强制要求有 'type' 字段。"""

    type: str


# --- 1. 数据变换 (Transform) 组件 ---
# class ResizeStepScalingConfig(ComponentConfig):
#     type: Literal["ResizeStepScaling"]
#     min_scale_factor: float
#     max_scale_factor: float
#     scale_step_size: float


class ResizeConfig(ComponentConfig):
    type: Literal["Resize"]
    size: list[int]


class RandomPaddingCropConfig(ComponentConfig):
    type: Literal["RandomPaddingCrop"]
    # crop_size: list[int]


class RandomHorizontalFlipConfig(ComponentConfig):
    type: Literal["RandomHorizontalFlip"]
    # 可以在这里添加 probability 等参数


class RandomDistortConfig(ComponentConfig):
    type: Literal["RandomDistort"]
    brightness_range: float
    contrast_range: float
    saturation_range: float


class NormalizeConfig(ComponentConfig):
    type: Literal["Normalize"]


class ToTensorConfig(ComponentConfig):
    type: Literal["ToTensor"]


# 使用 Union 来表示任何一种 Transform 配置
TransformConfig = Union[
    ResizeConfig,
    RandomPaddingCropConfig,
    RandomHorizontalFlipConfig,
    RandomDistortConfig,
    NormalizeConfig,
    ToTensorConfig,
]


# --- 2. 模型 (Model) 组件 ---
class BackboneConfig(ComponentConfig):
    pretrained: Optional[str] = None


class HeadConfig(ComponentConfig):
    num_classes: int


class ModelConfig(ComponentConfig):
    type: Literal["HRNetSegmentor"]  # 指定架构类型
    backbone: BackboneConfig
    head: HeadConfig
    # 允许架构自身有额外的参数
    # bb_channels: Optional[int] = None
    drop_prob: Optional[float] = None
    proj_dim: Optional[int] = None


# --- 3. 损失函数 (Loss) 组件 ---
class CrossEntropyLossConfig(ComponentConfig):
    type: Literal["CrossEntropyLoss"]
    ignore_index: int = 255


class PixelContrastCrossEntropyLossConfig(ComponentConfig):
    type: Literal["PixelContrastCrossEntropyLoss"]
    temperature: float = 0.1
    base_temperature: float = 0.07
    ignore_index: int = 255
    max_samples: int = 1024
    max_views: int = 100


# 使用 Union 来表示任何一种单一的损失配置
SingleLossConfig = Union[CrossEntropyLossConfig, PixelContrastCrossEntropyLossConfig]


# 复合损失的配置
class CombinedLossConfig(BaseModel):
    # 注意：这里我们不使用 ComponentConfig，因为它没有 `type` 字段
    types: list[SingleLossConfig]
    coef: list[float]


# --- 4. 优化器 (Optimizer) 组件 ---
class SGDConfig(ComponentConfig):
    type: Literal["SGD"]
    momentum: float = 0.9
    weight_decay: float = 0.0002


class AdamConfig(ComponentConfig):
    type: Literal["Adam"]
    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0


class AdamWConfig(ComponentConfig):
    type: Literal["AdamW"]
    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01


# 支持联合类型，方便配置文件解析
OptimizerConfig = Union[SGDConfig, AdamConfig, AdamWConfig]


# --- 5. 学习率调度器 (LR Scheduler) 组件 ---
class PolynomialDecayConfig(ComponentConfig):
    type: Literal["PolynomialDecay"]
    learning_rate: float
    end_lr: float = 0.0
    power: float = 0.9


LR_SchedulerConfig = Union[PolynomialDecayConfig]


# --- 6. 数据集 (Dataset) 组件 ---
class DatasetDetailConfig(BaseModel):
    type: str
    root_path: str
    transforms: list[TransformConfig]


# --- 7. 环境配置 (Environment) ---
class EnvironmentConfig(BaseModel):
    output_dir: str
    seed: Optional[int] = None
    num_workers: int = 4
    log_interval: int = 50
    save_interval: int = 1000


# --- 顶层配置结构 ---
class AppConfig(BaseModel):
    environment: EnvironmentConfig
    batch_size: int
    iters: int
    dataset: dict[str, DatasetDetailConfig]
    model: ModelConfig
    loss: CombinedLossConfig
    optimizer: OptimizerConfig
    lr_scheduler: LR_SchedulerConfig
