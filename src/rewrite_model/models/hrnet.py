import logging
import os
from dataclasses import dataclass, field
from typing import Type, List
import click
from omegaconf import DictConfig, OmegaConf
from rich import print
import requests

# from sympy import false
import torch.nn as nn
from torch.hub import load_state_dict_from_url


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ["hrnet18", "hrnet32", "hrnet48"]


class BasicBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        stride=1,
        downsample=None,
        momentum=0.1,
        relu_inplace=True,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(output_channels, momentum=momentum)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(output_channels, momentum=momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        input_channels,
        output_channels,
        stride=1,
        downsample=None,
        momentum=0.1,
        relu_inplace=True,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(output_channels, momentum=momentum)
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(output_channels, momentum=momentum)
        self.conv3 = nn.Conv2d(
            output_channels, output_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(output_channels * self.expansion, momentum=momentum)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)
        output = self.conv3(output)
        output = self.bn3(output)
        if self.downsample is not None:
            residual = self.downsample(x)
        output += residual
        output = self.relu(output)
        return output


blocks_dict = {
    "BASIC": BasicBlock,
    "BOTTLENECK": Bottleneck,
}


@dataclass
class HighResolutionModule(nn.Module):
    num_branches: int
    blocks: Type[nn.Module]
    num_blocks: List[int]
    input_channels: List[int]
    output_channels: List[int]
    fuse_method: str
    multi_scale_output: bool = True
    upsample_mode: str = "bilinear"
    momentum: float = 0.1
    relu_inplace: bool = False
    align_corners: bool = False
    branches: nn.ModuleList = field(init=False)
    fuse_layers: List[nn.ModuleList] = field(init=False)
    relu: nn.Module = field(init=False)

    def __post_init__(self):
        # 调用父类的构造函数
        super(HighResolutionModule, self).__init__()

        # 检查分支数、块数、输入通道数和输出通道数是否符合要求
        self._check_branches(
            self.num_branches,
            self.num_blocks,
            self.input_channels,
        )

        # 创建分支
        self.branches = self._make_branches(
            self.num_branches, self.blocks, self.num_blocks
        )

        # 创建融合层
        self.fuse_layers = self._make_fuse_layers()

        # 创建ReLU激活函数
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(
        self, num_branches: int, num_blocks: List[int], input_channels: List[int]
    ) -> None:
        """
        检查分支数量、块数量和输入通道数是否匹配。

        Args:
            num_branches (int): 分支数量。
            num_blocks (list of int): 每个分支的块数量列表。
            input_channels (list of int): 每个分支的输入通道数列表。

        Raises:
            ValueError: 如果分支数量与块数量列表长度、输入通道数列表长度不匹配。
        """
        if num_branches != len(num_blocks):
            raise ValueError(
                f"分支数量({num_branches})与块数量列表长度({len(num_blocks)})不匹配。"
            )
        if num_branches != len(input_channels):
            raise ValueError(
                f"分支数量({num_branches})与输入通道数列表长度({len(input_channels)})不匹配。"
            )

    def _make_one_branch(self, branch_index, block, num_blocks, stride=1):
        """
        创建一个分支的网络结构。

        Args:
            branch_index (int): 分支的索引。
            block (nn.Module): 用于构建分支的基本块。
            num_blocks (List[int]): 每个分支中基本块的数量。
            stride (int, optional): 卷积的步长，默认为1。

        Returns:
            nn.Sequential: 包含所有基本块的Sequential模块。

        """
        downsample = None
        if (
            stride != 1
            or self.input_channels[branch_index] != self.output_channels[branch_index]
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.input_channels[branch_index],
                    self.output_channels[branch_index],
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    self.output_channels[branch_index], momentum=self.momentum
                ),
            )
        layers = []
        layers.append(
            block(
                self.input_channels[branch_index],
                self.output_channels[branch_index],
                stride,
                downsample,
            )
        )
        # self.input_channels[branch_index]=self.output_channels[branch_index]
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.output_channels[branch_index],
                    self.output_channels[branch_index],
                )
            )
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks):
        """
        生成多个分支。

        Args:
            num_branches (int): 分支的数量。
            block (nn.Module): 用于构建分支的模块。
            num_blocks (int): 每个分支中模块的数量。

        Returns:
            nn.ModuleList: 包含所有分支的列表。

        """
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """
        根据网络分支数量创建融合层。

        Args:
            无

        Returns:
            list: 包含融合层的列表，每个元素对应一个输出尺度的融合层列表。

        """
        fuse_layers = []
        if self.num_branches == 1:
            return fuse_layers
        for i in range(self.num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    # TODO: upsample place in forward
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                self.output_channels[j],
                                self.output_channels[i],
                                1,
                                1,
                                0,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                self.output_channels[j], momentum=self.momentum
                            ),
                            nn.Upsample(
                                scale_factor=2 ** (j - i),
                                mode=self.upsample_mode,
                                align_corners=self.align_corners,
                            ),
                        )
                    )
                if j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        self.output_channels[i - 1],
                                        self.output_channels[i],
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        self.output_channels[i], momentum=self.momentum
                                    ),
                                )
                            )
                        else:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        self.output_channels[k + j],
                                        self.output_channels[k + j + 1],
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        self.output_channels[k + j + 1],
                                        momentum=self.momentum,
                                    ),
                                    nn.ReLU(inplace=self.relu_inplace),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return fuse_layers

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (list of Tensor): 输入的特征图列表，每个元素对应于一个分支的输入。

        Returns:
            list of Tensor: 融合后的特征图列表，每个元素对应于一个分支的输出。

        """
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i, fuse_layer in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fuse_layer[0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y += x[j]
                else:
                    y += fuse_layer[j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HighResolutionNet(nn.Module):
    """
    这是一个HighResolutionNet类的文档字符串。

    该类是用于实现高分辨率网络的模型，继承自nn.Module。
    """

    def __init__(self, cfg: DictConfig):
        """
        初始化 HighResolutionNet 类。

        Args:
            cfg (DictConfig): 配置文件，包含网络结构参数等。

        Returns:
            None

        Raises:
            ValueError: 如果配置文件中不包含 'STAGE1', 'STAGE2', 'STAGE3', 'STAGE4' 字段之一，则抛出异常。

        """
        super(HighResolutionNet, self).__init__()
        self.cfg = cfg

        required_keys = {"STAGE1", "STAGE2", "STAGE3", "STAGE4"}
        missing_keys = required_keys - cfg.keys()
        if missing_keys:
            raise ValueError(
                f"Configuration is missing required fields: \
                {', '.join(missing_keys)}"
            )

        self.stage1 = self._make_layer(cfg.STAGE1)
        self.stage2 = self._make_stage(cfg.STAGE2)
        self.stage3 = self._make_stage(cfg.STAGE3)
        self.stage4 = self._make_stage(cfg.STAGE4)

    def _make_layer(self, layer_cfg: DictConfig):
        """
        创建层结构。

        Args:
            layer_cfg (DictConfig): 配置字典，包含转换层的通道数等信息。

        Returns:
            nn.ModuleList: 包含两个层的nn.ModuleList对象，每个层由多个卷积层和瓶颈层组成，
                并在最后通过一个1x1卷积层将通道数转换为配置中指定的通道数。

        """
        momentum = self.cfg.BASE.BN_MOMENTUM
        relu_inplace = self.cfg.BASE.RELU_INPLACE
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, momentum=momentum),
        )
        layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=momentum),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=momentum),
            nn.ReLU(inplace=relu_inplace),
            Bottleneck(
                64,
                64,
                stride=1,
                momentum=momentum,
                relu_inplace=relu_inplace,
                downsample=downsample,
            ),
            Bottleneck(
                256,
                64,
                stride=1,
                momentum=momentum,
                relu_inplace=relu_inplace,
            ),
            Bottleneck(
                256,
                64,
                stride=1,
                momentum=momentum,
                relu_inplace=relu_inplace,
            ),
            Bottleneck(
                256,
                64,
                stride=1,
                momentum=momentum,
                relu_inplace=relu_inplace,
            ),
        )
        layer1 = nn.Sequential(
            layer,
            nn.Conv2d(256, layer_cfg.OUTPUT_CHANNELS[0], kernel_size=1, bias=False),
        )
        layer2 = nn.Sequential(
            layer,
            nn.Conv2d(256, layer_cfg.OUTPUT_CHANNELS[1], kernel_size=1, bias=False),
        )
        return nn.ModuleList([layer1, layer2])

    def _make_stage(self, stage_cfg: DictConfig):
        """
        根据配置字典创建阶段模块

        Args:
            stage_cfg (DictConfig): 阶段配置字典，包含以下字段：
                NUM_MODULES (int): 阶段中模块的数量
                BLOCK (str): 块类型，如 'BottleneckWithFixedBatchNorm'
                NUM_BLOCKS (int): 每个分支中的块数量
                NUM_BRANCHES (int): 分支数量
                INPUT_CHANNELS (int): 输入通道数
                OUTPUT_CHANNELS (int): 输出通道数
                FUSE_METHOD (str): 特征融合方法，如 'SUM'
                MULTI_SCALE_OUTPUT (bool, optional): 是否输出多尺度特征图，默认为 False

        Returns:
            nn.Sequential: 阶段模块，由多个 HighResolutionModule 组成

        """
        num_modules = stage_cfg["NUM_MODULES"]
        block = blocks_dict[stage_cfg["BLOCK"]]
        num_blocks = stage_cfg["NUM_BLOCKS"]
        num_branches = stage_cfg["NUM_BRANCHES"]
        input_channels = stage_cfg["INPUT_CHANNELS"]
        output_channels = stage_cfg["OUTPUT_CHANNELS"]
        fuse_method = stage_cfg["FUSE_METHOD"]
        multi_scale_output = stage_cfg.get("MULTI_SCALE_OUTPUT", False)
        momentum = self.cfg.BASE.BN_MOMENTUM
        relu_inplace = self.cfg.BASE.RELU_INPLACE
        align_corners = self.cfg.BASE.ALIGN_CORNERS

        modules = []
        for _ in range(num_modules - 1):
            modules.append(
                HighResolutionModule(
                    num_branches=num_branches,
                    blocks=block,
                    num_blocks=num_blocks,
                    input_channels=input_channels,
                    output_channels=input_channels,
                    fuse_method=fuse_method,
                    multi_scale_output=False,
                    momentum=momentum,
                    relu_inplace=relu_inplace,
                    align_corners=align_corners,
                )
            )
        modules.append(
            HighResolutionModule(
                num_branches=num_branches,
                blocks=block,
                num_blocks=num_blocks,
                input_channels=input_channels,
                output_channels=output_channels,
                fuse_method=fuse_method,
                multi_scale_output=multi_scale_output,
                momentum=momentum,
                relu_inplace=relu_inplace,
                align_corners=align_corners,
            )
        )
        return nn.Sequential(*modules)

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入的Tensor数据。

        Returns:
            Tensor: 经过所有stage处理后的输出Tensor。

        """
        x = [stage(x) for stage in self.stage1]
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


def _hrnet(config_path: str, pretrained: bool, **kwargs: dict) -> HighResolutionNet:
    """
    创建并返回HighResolutionNet模型的实例。

    Args:
        config_path (str): 配置文件的路径。
        pretrained (bool): 是否加载预训练权重。
        **kwargs (dict): 传递给HighResolutionNet构造函数的额外关键字参数。

    Returns:
        HighResolutionNet: 创建好的HighResolutionNet模型实例。

    Raises:
        ValueError: 如果配置文件无效、模型实例创建失败或预训练权重加载失败，则抛出异常。
    """
    try:
        # 加载配置
        cfg = OmegaConf.load(config_path)  # 检查路径是否存在
        # 如果存在_base_，则递归地加载和合并基础配置
        if "_base_" in cfg:
            base_configs = cfg._base_

            # 如果_base_指向多个文件，则依次合并
            if isinstance(base_configs, list):
                for base_config in base_configs:
                    # 解析相对路径
                    base_config_path = os.path.join(
                        os.path.dirname(config_path), base_config
                    )
                    base_cfg = OmegaConf.load(base_config_path)
                    cfg = OmegaConf.merge(base_cfg, cfg)
            else:
                # 解析相对路径
                base_config_path = os.path.join(
                    os.path.dirname(config_path), base_configs
                )
                base_cfg = OmegaConf.load(base_config_path)
                cfg = OmegaConf.merge(base_cfg, cfg)
        logger.info(
            "Configuration loaded from: %s, Configuration content: %s", config_path, cfg
        )
        # 合并 kwargs 到 cfg 中
        cfg = OmegaConf.merge(cfg, kwargs)
        logger.info("Merged configuration: %s", kwargs)
        assert isinstance(
            cfg, DictConfig
        ), f"cfg 必须是 DictConfig 类型, 实际类型为 {type(cfg)}"
        # 冻结配置
        OmegaConf.set_readonly(cfg, True)
        logger.info("Configuration frozen")
        # 创建模型实例
        model = HighResolutionNet(cfg=cfg)
        logger.info("Model instance created: %s", model.__class__.__name__)
        # 如果需要，加载预训练权重
        if pretrained:
            try:
                model_url = cfg.get("model_url", None)

                def validate_url(url: str) -> bool:
                    """验证URL是否有效"""
                    try:
                        response = requests.head(url, timeout=5)
                        return response.status_code == 200
                    except requests.RequestException:
                        return False

                if model_url is None or not validate_url(model_url):
                    raise ValueError(
                        "Pretrained model URL is not specified or \
                        invalid in the configuration."
                    )
                logger.info("Loading pretrained weights from: %s", model_url)
                state_dict = load_state_dict_from_url(model_url, map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
                logger.info("Pretrained weights loaded successfully.")
            except Exception as e:
                logger.error("Error loading pretrained weights: %s", e)
                raise ValueError("Pretrained weights loading failed") from e
        return model
    except FileNotFoundError as e:
        logger.error("Configuration file not found at path: %s", config_path)
        raise ValueError("Failed to load configuration file") from e
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        raise ValueError("Failed to create model instance") from e


def hrnet18(pretrained: bool = True, **kwargs: dict) -> HighResolutionNet:
    """
    加载预训练的HRNet18模型。

    Args:
        pretrained (bool, optional): 是否加载预训练模型权重，默认为True。
        **kwargs (dict, optional): 其他关键字参数，将直接传递给 _hrnet 函数。

    Returns:
        HighResolutionNet: 加载好的HRNet18模型实例。

    """
    config_path = "../config/model/hrnet18_config.yaml"
    config_path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
    )
    return _hrnet(config_path, pretrained, **kwargs)


def hrnet32(pretrained: bool = True, **kwargs: dict) -> HighResolutionNet:
    """
    构建HRNet-32模型。

    Args:
        pretrained (bool, optional): 是否加载预训练权重，默认为True。
        **kwargs (dict, optional): 其他关键字参数，这些参数将传递给_hrnet函数。

    Returns:
        HighResolutionNet: 构建好的HRNet-32模型。

    """
    config_path = "../config/model/hrnet32_config.yaml"
    return _hrnet(config_path, pretrained, **kwargs)


def hrnet48(pretrained: bool = True, **kwargs: dict) -> HighResolutionNet:
    """
    创建 HRNet48 模型实例。

    Args:
        pretrained (bool, optional): 是否加载预训练模型。默认为 True。
        **kwargs (dict, optional): 传递给 _hrnet 函数的其他关键字参数。

    Returns:
        HighResolutionNet: 返回创建好的 HRNet48 模型实例。

    """
    config_path = "../config/model/hrnet48_config.yaml"
    return _hrnet(config_path, pretrained, **kwargs)
