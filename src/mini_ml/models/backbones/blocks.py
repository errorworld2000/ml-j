from typing import Callable, Optional

import torch
from torch import nn

__all__ = ["ConvBNReLU", "BasicBlock", "BottleneckBlock"]


class ConvBNReLU(nn.Sequential):
    """
    一个标准的卷积块，包含了 Conv2d, BatchNorm2d, 和可选的 ReLU 激活函数。
    这大大减少了在定义网络层时的代码重复。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int, optional): 卷积核大小。默认为 3。
        stride (int, optional): 卷积步长。默认为 1。
        padding (int, optional): 填充大小。如果为 None，则根据 kernel_size 自动计算。默认为 None。
        groups (int, optional): 分组卷积的组数。默认为 1。
        bias (bool, optional): Conv2d 层是否使用偏置。BatchNorm 存在时通常设为 False。默认为 False。
        norm_layer (Callable[..., nn.Module], optional): 使用的归一化层。默认为 nn.BatchNorm2d。
        activation_layer (Callable[..., nn.Module], optional): 使用的激活函数。如果为 None，则不添加激活层。默认为 nn.ReLU。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        bias: bool = False,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
    ) -> None:
        if padding is None:
            # 自动计算 padding 以保持特征图尺寸不变 (当 stride=1)
            padding = (kernel_size - 1) // 2

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=bias,
            ),
            norm_layer(out_channels),
        ]
        if activation_layer is not None:
            layers.append(activation_layer())

        super().__init__(*layers)


class BasicBlock(nn.Module):
    """
    ResNet 的基础残差块 (BasicBlock)。
    对应于 ResNet-18/34。
    """

    expansion: int = 1  # 输出通道相对于 planes 的扩展因子

    def __init__(
        self,
        input_channels: int,
        planes: int,  # 内部卷积层的通道数
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()

        self.conv1 = ConvBNReLU(
            input_channels, planes, kernel_size=3, stride=stride, norm_layer=norm_layer
        )
        self.conv2 = ConvBNReLU(
            planes, planes, kernel_size=3, activation_layer=None, norm_layer=norm_layer
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    """
    ResNet 的瓶颈残差块 (BottleneckBlock)。
    对应于 ResNet-50/101/152。
    """

    expansion: int = 4  # 输出通道相对于 planes 的扩展因子

    def __init__(
        self,
        input_channels: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()

        width = int(planes * (base_width / 64.0)) * groups

        # 主路径：1x1 -> 3x3 -> 1x1
        self.conv1 = ConvBNReLU(
            input_channels, width, kernel_size=1, norm_layer=norm_layer
        )
        self.conv2 = ConvBNReLU(
            width,
            width,
            kernel_size=3,
            stride=stride,
            groups=groups,
            norm_layer=norm_layer,
        )
        self.conv3 = ConvBNReLU(
            width,
            planes * self.expansion,
            kernel_size=1,
            activation_layer=None,
            norm_layer=norm_layer,
        )

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        if self.downsample:
            residual = self.downsample(x)
        output += residual
        output = self.relu(output)
        return output
