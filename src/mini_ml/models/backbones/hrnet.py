import logging

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn

from mini_ml.utils.register import BACKBONES

from mini_ml.models.backbones.blocks import BasicBlock, BottleneckBlock, ConvBNReLU

logger = logging.getLogger(__name__)
logger.info("Loading HRNet model")

__all__ = ["HRNet_W18", "HRNet_W32", "HRNet_W48", "HRNet_W64"]


class StageConfig(BaseModel):
    """Stage 结构"""

    num_modules: int
    num_blocks: list[int]
    num_channels: list[int]


class HRNetConfig(BaseModel):
    """HRNet 结构"""

    stage1: StageConfig
    stage2: StageConfig
    stage3: StageConfig
    stage4: StageConfig


class HRNet(nn.Module):
    """
    高分辨率网络 (High-Resolution Network)。

    这是一个功能强大的语义分割主干网络，其核心特点是在整个网络中
    持续维持高分辨率的特征图，并通过重复的多尺度融合来交换信息。
    """

    def __init__(
        self,
        config: HRNetConfig,
        input_channels: int = 3,
        # Squeeze-and-Excitation, Polarized Self-Attention 等目前未使用，作为保留参数
        has_se: bool = False,
        use_psa: bool = False,
        align_corners: bool = False,
    ):
        super().__init__()
        self.align_corners = align_corners

        # --- 1. Stem (初始降采样层) ---
        # 两个步长为 2 的卷积，将分辨率降低 4 倍
        self.stem = nn.Sequential(
            ConvBNReLU(input_channels, 64, kernel_size=3, stride=2),
            ConvBNReLU(64, 64, kernel_size=3, stride=2),
        )

        # --- 2. Stage 1 ---
        s1_cfg = config.stage1
        self.layer1 = Layer1(
            input_channels=64,
            output_channels=s1_cfg.num_channels[0],
            num_blocks=s1_cfg.num_blocks[0],
        )

        # --- 3. 循环创建后续的 Stages 和 Transitions ---
        self.transitions = nn.ModuleList()
        self.stages = nn.ModuleList()
        prev_stage_channels = [s1_cfg.num_channels[0] * BottleneckBlock.expansion]
        for i in range(2, 5):
            stage_key = f"stage{i}"
            stage_cfg: StageConfig = getattr(config, stage_key)
            current_stage_channels = stage_cfg.num_channels

            # 创建 Transition Layer
            transition = TransitionLayer(
                input_channels=prev_stage_channels,
                output_channels=current_stage_channels,
            )
            self.transitions.append(transition)

            # 创建 Stage
            stage = Stage(
                num_modules=stage_cfg.num_modules,
                input_channels=current_stage_channels,
                output_channels=current_stage_channels,
                num_blocks=stage_cfg.num_blocks,
                align_corners=align_corners,
            )
            self.stages.append(stage)
            prev_stage_channels = current_stage_channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        x = [x]
        for transition, stage in zip(self.transitions, self.stages):
            x = transition(x)
            x = stage(x)
        # 这是 HRNet 作为分割主干网络的标准输出
        hr_feat = x[0]
        hr_size = hr_feat.shape[-2:]
        # 使用列表推导式对其他分辨率的特征图进行上采样
        interpolated_feats = [
            F.interpolate(
                branch_feat,
                size=hr_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            for branch_feat in x[1:]
        ]

        # 拼接所有特征图
        output = torch.cat([hr_feat] + interpolated_feats, dim=1)

        # 通常主干网络会返回一个包含单个元素的列表
        return [output]

    def load_pretrained(self, path: str):
        """一个专门用于加载预训练权重的辅助函数"""
        try:
            state_dict = torch.load(path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded pretrained weights from {path}")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")


class Branches(nn.Module):
    """
    为 HRNet 的每个并行分支构建一系列的 BasicBlock。
    """

    def __init__(
        self,
        num_blocks: list[int],
        input_channels: list[int],
        output_channels: list[int],
    ):
        super().__init__()
        self.branches = nn.ModuleList()

        for i, out_channel in enumerate(output_channels):
            sublist = []
            for j in range(num_blocks[i]):
                in_ch = input_channels[i] if j == 0 else out_channel
                sublist.append(
                    BasicBlock(
                        input_channels=in_ch,
                        planes=output_channels[i],
                    )
                )
            self.branches.append(nn.Sequential(*sublist))

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        assert len(x) == len(
            self.block_list
        ), f"Expected {len(self.block_list)} inputs, but got {len(x)}"
        outs = [branch(branch_input) for branch, branch_input in zip(self.branches, x)]

        return outs


class FuseLayer(nn.Module):
    """
    HRNet 的多分辨率融合层。

    对于每一个目标输出分支 `i`，它会融合所有输入分支 `j` 的信息：
    - 当 j > i (输入分辨率更低): 通过 '1x1 Conv -> BatchNorm -> Upsample' 进行上采样融合。
    - 当 j == i (输入分辨率相同): 直接恒等映射 (Identity)。
    - 当 j < i (输入分辨率更高): 通过一系列步进卷积进行下采样融合。
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        multi_scale_output: bool = True,
        align_corners: bool = False,
    ):
        super().__init__()

        self.fuse_layers = nn.ModuleList()
        self.num_in_branches = len(in_channels)
        self.num_out_branches = len(out_channels) if multi_scale_output else 1
        self.align_corners = align_corners

        for i, out_channel in enumerate(out_channels):
            # 每个输出分支都需要一个融合层列表，用于处理来自所有输入的连接
            fuse_layer_for_output_i = nn.ModuleList()
            for j, in_channel in enumerate(in_channels):
                if j > i:
                    fuse_layer_for_output_i.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channel, out_channel, kernel_size=1, bias=False
                            ),
                            nn.BatchNorm2d(out_channel),
                        )
                    )
                elif j == i:
                    fuse_layer_for_output_i.append(nn.Identity())
                elif j < i:
                    sequence = []
                    for k in range(i - j):
                        in_ch = in_channel if k == 0 else out_channels[j]
                        out_ch = out_channel if k == i - j - 1 else out_channels[j]

                        sequence.append(
                            ConvBNReLU(
                                in_ch,
                                out_ch,
                                kernel_size=3,
                                stride=2,
                            )
                        )
                    fuse_layer_for_output_i.append(nn.Sequential(*sequence))

            self.fuse_layers.append(fuse_layer_for_output_i)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        outs = []

        for i in range(self.num_out_branches):
            y = x[i]
            for j in enumerate(self.num_in_branches):
                if j == i:
                    pass

                output = self.fuse_layers[i][j](x[j])
                if j > i:
                    output = F.interpolate(
                        output,
                        size=y.shape[-2:],
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                y = y + output

            outs.append(F.relu(y, inplace=True))

        return outs


class HighResolutionModule(nn.Module):
    """
    HRNet 的核心模块 (HRModule)。

    它由两个主要部分组成：
    1. 并行分支 (Branches): 在每个分辨率上独立地应用一系列残差块。
    2. 融合层 (FuseLayer): 在所有分辨率之间交换和融合信息。
    """

    def __init__(
        self,
        input_channels: list[int],
        output_channels: list[int],
        num_blocks: list[int],
        multi_scale_output: bool = True,
        align_corners: bool = False,
    ):
        super().__init__()

        self.branches = Branches(
            input_channels=input_channels,
            output_channels=output_channels,
            num_blocks=num_blocks,
        )
        self.fuse_layer = FuseLayer(
            in_channels=output_channels,
            out_channels=output_channels,
            multi_scale_output=multi_scale_output,
            align_corners=align_corners,
        )

    def forward(self, x):
        return self.fuse_layer(self.branches(x))


class Layer1(nn.Module):
    """
    HRNet 的第一个主要层，功能上等同于 ResNet 的 stage 1。
    它由一系列的 BottleneckBlock 组成。
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_blocks: int,
    ):
        super().__init__()

        self.block_list = nn.Sequential()

        downsample = nn.Sequential(
            ConvBNReLU(
                input_channels,
                output_channels * BottleneckBlock.expansion,
                kernel_size=1,
                activation_layer=None,
            )
        )

        for i in range(num_blocks):
            bottleneck_block = BottleneckBlock(
                input_channels=(
                    input_channels
                    if i == 0
                    else output_channels * BottleneckBlock.expansion
                ),
                planes=output_channels,
                downsample=downsample if i == 0 else None,
            )
            self.block_list.append(bottleneck_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block_list(x)


class Stage(nn.Module):
    """
    HRNet 的一个 Stage，由一个或多个 HighResolutionModule 堆叠而成。
    """

    def __init__(
        self,
        num_modules: int,
        input_channels: list[int],
        output_channels: list[int],
        num_blocks: list[int],
        multi_scale_output: bool = True,
        align_corners: bool = False,
    ):
        super().__init__()

        self.module_list = nn.ModuleList()
        for i in range(num_modules):
            multi_scale = not (i == num_modules - 1 and not multi_scale_output)
            self.module_list.append(
                HighResolutionModule(
                    input_channels=input_channels,
                    output_channels=output_channels,
                    num_blocks=num_blocks,
                    multi_scale_output=multi_scale,
                    align_corners=align_corners,
                )
            )
            input_channels = output_channels  # 更新输入通道数为当前输出通道数

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        for module in self.module_list:
            x = module(x)
        return x


class TransitionLayer(nn.Module):
    """
    用于在 HRNet 的不同 Stage 之间进行过渡。

    它接收一个多分辨率特征图列表，并输出一个新的列表，其中：
    1. 已有分辨率的特征图通道数被调整到目标数量。
    2. 新增一个通过对最低分辨率输入进行步进卷积而产生的更低分辨率的特征图。
    """

    def __init__(
        self,
        input_channels: list[int],
        output_channels: list[int],
    ):
        super().__init__()

        # 确保输出分支比输入多一个（这是 HRNet 的设计）
        assert (
            len(output_channels) == len(input_channels) + 1
        ), "TransitionLayer must create one new resolution branch."

        self.conv_list = nn.ModuleList()
        for i, out_channel in enumerate(output_channels):
            if i < len(input_channels):
                in_ch = input_channels[i]
                if in_ch != out_channel:
                    self.conv_list.append(
                        ConvBNReLU(
                            in_ch,
                            out_channel,
                            kernel_size=3,
                        )
                    )
                else:
                    self.conv_list.append(nn.Identity())
            else:
                last_in_ch = input_channels[-1]
                self.conv_list.append(
                    ConvBNReLU(
                        last_in_ch,
                        out_channel,
                        kernel_size=3,
                        stride=2,
                    )
                )

    def forward(self, x):
        outs = []
        for idx, conv in enumerate(self.conv_list):
            if idx < len(x):
                outs.append(conv(x[idx]))
            else:
                outs.append(conv(x[-1]))
        return outs

class SimpleSegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        
        # 使用一个 1x1 卷积将特征图的通道数转换为类别数
        # 每个输出通道对应一个类别的得分图
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features 是 HRNet backbone 输出的拼接后的特征图
        return self.conv(features)

HRNET_RAW_CONFIGS: dict[str, dict[str, any]] = {
    "W18": {
        "stage1": {"num_modules": 1, "num_blocks": [4], "num_channels": [64]},
        "stage2": {"num_modules": 1, "num_blocks": [4, 4], "num_channels": [18, 36]},
        "stage3": {
            "num_modules": 4,
            "num_blocks": [4, 4, 4],
            "num_channels": [18, 36, 72],
        },
        "stage4": {
            "num_modules": 3,
            "num_blocks": [4, 4, 4, 4],
            "num_channels": [18, 36, 72, 144],
        },
    },
    "W32": {
        "stage1": {"num_modules": 1, "num_blocks": [4], "num_channels": [64]},
        "stage2": {"num_modules": 1, "num_blocks": [4, 4], "num_channels": [32, 64]},
        "stage3": {
            "num_modules": 4,
            "num_blocks": [4, 4, 4],
            "num_channels": [32, 64, 128],
        },
        "stage4": {
            "num_modules": 3,
            "num_blocks": [4, 4, 4, 4],
            "num_channels": [32, 64, 128, 256],
        },
    },
    "W48": {
        "stage1": {"num_modules": 1, "num_blocks": [4], "num_channels": [64]},
        "stage2": {"num_modules": 1, "num_blocks": [4, 4], "num_channels": [48, 96]},
        "stage3": {
            "num_modules": 4,
            "num_blocks": [4, 4, 4],
            "num_channels": [48, 96, 192],
        },
        "stage4": {
            "num_modules": 3,
            "num_blocks": [4, 4, 4, 4],
            "num_channels": [48, 96, 192, 384],
        },
    },
    "W64": {
        "stage1": {"num_modules": 1, "num_blocks": [4], "num_channels": [64]},
        "stage2": {"num_modules": 1, "num_blocks": [4, 4], "num_channels": [64, 128]},
        "stage3": {
            "num_modules": 4,
            "num_blocks": [4, 4, 4],
            "num_channels": [64, 128, 256],
        },
        "stage4": {
            "num_modules": 3,
            "num_blocks": [4, 4, 4, 4],
            "num_channels": [64, 128, 256, 512],
        },
    },
}


def _build_hrnet(
    model_name: str,  # 新增参数：要构建的模型名，如 'W18'
    pretrained: bool = False,
    **kwargs,
):
    """
    一个通用的 HRNet 构建器函数。
    它根据模型名称查找配置，验证配置，创建模型，并处理预训练权重。
    """
    # 1. 从配置中心获取原始配置字典
    raw_config = HRNET_RAW_CONFIGS.get(model_name)
    if raw_config is None:
        raise ValueError(
            f"Unknown model name: {model_name}. Available models: {list(HRNET_RAW_CONFIGS.keys())}"
        )

    # 2. 使用 Pydantic 验证配置
    try:
        validated_config = HRNetConfig(**raw_config)
    except Exception as e:
        print(f"Configuration validation failed for {model_name}: {e}")
        raise e

    # 3. 用验证后的配置创建模型
    model = HRNet(config=validated_config, **kwargs)

    if pretrained:
        print(f"INFO: Pretrained weights for HRNet_{model_name} should be loaded here.")
        # model.load_pretrained(f'path/to/hrnet_{model_name}_weights.pth')

    return model


@BACKBONES.register()
def HRNet_W18(pretrained: bool = False, **kwargs):
    """构建一个 HRNet-W18 模型。"""
    return _build_hrnet(model_name="W18", pretrained=pretrained, **kwargs)


@BACKBONES.register()
def HRNet_W32(pretrained: bool = False, **kwargs):
    """构建一个 HRNet-W32 模型。"""
    return _build_hrnet(model_name="W32", pretrained=pretrained, **kwargs)


@BACKBONES.register()
def HRNet_W48(pretrained: bool = False, **kwargs):
    """构建一个 HRNet-W48 模型。"""
    return _build_hrnet(model_name="W48", pretrained=pretrained, **kwargs)


@BACKBONES.register()
def HRNet_W64(pretrained: bool = False, **kwargs):
    """构建一个 HRNet-W64 模型。"""
    return _build_hrnet(model_name="W64", pretrained=pretrained, **kwargs)
