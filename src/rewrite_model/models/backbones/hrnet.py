import torch
from torch import nn
import torch.nn.functional as F

from rewrite_model.utils import logger, utils
from rewrite_model.utils.register import BACKBONES

logger.info("Loading HRNet model")

__all__ = ["HRNet_W18", "HRNet_W32", "HRNet_W48", "HRNet_W64"]


class HRNet(nn.Module):
    """_summary_

    Args:
        has_se (bool, optional): Whether to use Squeeze-and-Excitation module. Default False.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        use_psa (bool, optional): Usage of the polarized self attention moudle. Default False.
    """

    def __init__(
        self,
        pretrained=None,
        input_channels=3,
        stage1_num_modules=1,
        stage1_num_blocks=(4,),
        stage1_num_channels=(64,),
        stage2_num_modules=1,
        stage2_num_blocks=(
            4,
            4,
        ),
        stage2_num_channels=(18, 36),
        stage3_num_modules=4,
        stage3_num_blocks=(
            4,
            4,
            4,
        ),
        stage3_num_channels=(18, 36, 72),
        stage4_num_modules=3,
        stage4_num_blocks=(4, 4, 4, 4),
        stage4_num_channels=(18, 36, 72, 144),
        has_se=False,
        align_corners=False,
        padding_same=False,
        use_psa=False,
    ):
        super(HRNet, self).__init__()
        self.pretrained = pretrained
        self.align_corners = align_corners
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_channels,
                64,
                kernel_size=3,
                stride=2,
                padding="same" if padding_same else 1,
                bias=False,
            )
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                stride=2,
                padding="same" if padding_same else 1,
                bias=False,
            )
        )
        self.lay1 = Layer1(
            input_channels=64,
            num_blocks=stage1_num_blocks[0],
            output_channels=stage1_num_channels[0],
            name="layer1",
            padding_same=padding_same,
        )
        self.tr1 = TransitionLayer(
            input_channels=[stage1_num_channels[0] * 4],
            output_channels=stage2_num_channels,
            padding_same=padding_same,
            name="tr1",
        )
        self.stage2 = Stage(
            num_modules=stage2_num_modules,
            num_blocks=stage2_num_blocks,
            input_channels=stage2_num_channels,
            output_channels=stage2_num_channels,
            name="stage2",
            padding_same=padding_same,
            align_corners=align_corners,
        )
        self.tr2 = TransitionLayer(
            input_channels=stage2_num_channels,
            output_channels=stage3_num_channels,
            padding_same=padding_same,
            name="tr2",
        )
        self.stage3 = Stage(
            num_modules=stage3_num_modules,
            num_blocks=stage3_num_blocks,
            input_channels=stage3_num_channels,
            output_channels=stage3_num_channels,
            name="stage3",
            padding_same=padding_same,
            align_corners=align_corners,
        )
        self.tr3 = TransitionLayer(
            input_channels=stage3_num_channels,
            output_channels=stage4_num_channels,
            padding_same=padding_same,
            name="tr3",
        )
        self.stage4 = Stage(
            num_modules=stage4_num_modules,
            num_blocks=stage4_num_blocks,
            input_channels=stage4_num_channels,
            output_channels=stage4_num_channels,
            name="stage4",
            padding_same=padding_same,
            align_corners=align_corners,
        )
        if self.pretrained is not None:
            utils.load_pretrained_model(self, "model.safetensors")

    def forward(self, x):
        outer = self.conv1(x)
        outer = self.conv2(outer)
        outer = self.lay1(outer)
        outer = self.tr1([outer])
        outer = self.stage2(outer)
        outer = self.tr2(outer)
        outer = self.stage3(outer)
        outer = self.tr3(outer)
        outer = self.stage4(outer)
        size = outer[0].shape[2:]  # 获取 height 和 width
        x1 = F.interpolate(
            outer[1], size=size, mode="bilinear", align_corners=self.align_corners
        )
        x2 = F.interpolate(
            outer[2], size=size, mode="bilinear", align_corners=self.align_corners
        )
        x3 = F.interpolate(
            outer[3], size=size, mode="bilinear", align_corners=self.align_corners
        )
        x = torch.cat([outer[0], x1, x2, x3], dim=1)
        return [x]


class BasicBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        stride=1,
        downsample=False,
        padding_same=True,
        name=None,
    ):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding="same" if padding_same else 1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding="same" if padding_same else 1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.downsample = downsample
        if self.downsample:
            self.conv_down = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.conv_down(x)
        out += residual
        out = self.relu(out)
        return out


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        stride=1,
        downsample=False,
        padding_same=True,
        name=None,
    ):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding="same" if padding_same else 1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.conv3 = nn.Conv2d(
            output_channels, output_channels * 4, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(output_channels * 4)
        self.downsample = downsample
        if self.downsample:
            self.conv_down = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels * 4),
            )

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
        if self.downsample:
            residual = self.conv_down(x)
        output += residual
        output = self.relu(output)
        return output


class Branches(nn.Module):
    def __init__(
        self,
        num_blocks,
        input_channels,
        output_channels,
        padding_same=True,
        name=None,
    ):
        super(Branches, self).__init__()
        self.block_list = []
        for i, out_channel in enumerate(output_channels):
            sublist = nn.ModuleList()  # 子分支依然使用 ModuleList 管理
            for j in range(num_blocks[i]):
                in_ch = input_channels[i] if j == 0 else out_channel
                basic_block = BasicBlock(
                    input_channels=in_ch,
                    output_channels=out_channel,
                    name=(name + f"_branch_layer_{i + 1}_{j + 1}" if name else None),
                    padding_same=padding_same,
                )
                sublist.append(basic_block)
            self.block_list.append(sublist)

    def forward(self, x):
        assert len(x) == len(
            self.block_list
        ), f"Expected {len(self.block_list)} inputs, but got {len(x)}"
        outs = []
        for idx, branch_input in enumerate(x):  # 使用 enumerate
            output = branch_input
            for basic_block in self.block_list[idx]:  # 显式访问嵌套 ModuleList
                output = basic_block(output)
            outs.append(output)
        return outs


class TransitionLayer(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        padding_same=True,
        name=None,
    ):
        super(TransitionLayer, self).__init__()
        self.conv_list = nn.ModuleList()
        for i, out_channel in enumerate(output_channels):
            if i < len(input_channels):
                if input_channels[i] != out_channel:
                    self.conv_list.append(
                        nn.Sequential(
                            nn.Conv2d(
                                input_channels[i],
                                out_channel,
                                kernel_size=3,
                                padding="same" if padding_same else 1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(out_channel),
                            nn.ReLU(),
                        )
                    )
                else:
                    self.conv_list.append(nn.Identity())
            else:
                self.conv_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            input_channels[-1],
                            out_channel,
                            kernel_size=3,
                            stride=2,
                            padding="same" if padding_same else 1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU(),
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


class FuseLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        multi_scale_output=True,
        padding_same=True,
        align_corners=False,
        name=None,
    ):
        super(FuseLayer, self).__init__()
        self._actual_out_ch = len(out_channels) if multi_scale_output else 1
        self._in_channels = in_channels
        self.align_corners = align_corners
        self.conv_lists = []
        for i, out_channel in enumerate(out_channels):
            conv_list = nn.ModuleList()
            for j, in_channel in enumerate(in_channels):
                if j > i:
                    conv_list.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channel, out_channel, kernel_size=1, bias=False
                            ),
                            nn.BatchNorm2d(out_channel),
                        )
                    )
                elif j == i:
                    conv_list.append(nn.Identity())
                elif j < i:
                    pre_channel = in_channel
                    sequence = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            sequence.append(
                                nn.Conv2d(
                                    pre_channel,
                                    out_channel,
                                    kernel_size=3,
                                    stride=2,
                                    padding="same" if padding_same else 1,
                                    bias=False,
                                )
                            )
                            sequence.append(nn.BatchNorm2d(out_channel))
                        else:
                            sequence.append(
                                nn.Conv2d(
                                    pre_channel,
                                    out_channels[j],
                                    kernel_size=3,
                                    stride=2,
                                    padding="same" if padding_same else 1,
                                    bias=False,
                                ),
                            )
                            sequence.append(nn.BatchNorm2d(out_channels[j]))
                            sequence.append(nn.ReLU())
                            pre_channel = out_channels[j]
                    conv_list.append(nn.Sequential(*sequence))

            self.conv_lists.append(conv_list)

    def forward(self, x):
        outs = []
        for i in range(self._actual_out_ch):
            residual = x[i]
            residual_shape = residual.shape[-2:]
            for j, in_channel in enumerate(self._in_channels):
                if j > i:
                    output = self.conv_lists[i][j](x[j])
                    output = F.interpolate(
                        output,
                        size=residual_shape,
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                    residual = residual + output
                elif j == i:
                    pass
                elif j < i:
                    output = self.conv_lists[i][j](x[j])
                    residual = residual + output
            #  add relu at last
            residual = F.relu(residual)
            outs.append(residual)
        return outs


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        num_blocks,
        multi_scale_output=True,
        padding_same=True,
        align_corners=False,
        name=None,
    ):
        super(HighResolutionModule, self).__init__()
        self.branches = Branches(
            input_channels=input_channels,
            output_channels=output_channels,
            num_blocks=num_blocks,
            padding_same=padding_same,
            name=name,
        )
        self.fuse_layer = FuseLayer(
            in_channels=output_channels,
            out_channels=output_channels,
            multi_scale_output=multi_scale_output,
            name=name,
            align_corners=align_corners,
            padding_same=padding_same,
        )

    def forward(self, x):
        return self.fuse_layer(self.branches(x))


class Layer1(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        num_blocks,
        padding_same=True,
        name=None,
    ):
        super(Layer1, self).__init__()
        self.block_list = nn.Sequential()

        for i in range(num_blocks):
            bottleneck_block = BottleneckBlock(
                input_channels=input_channels if i == 0 else output_channels * 4,
                output_channels=output_channels,
                stride=1,
                downsample=True if i == 0 else False,
                name=name + "_" + str(i + 1) if name else None,
                padding_same=padding_same,
            )
            self.block_list.append(bottleneck_block)

    def forward(self, x):
        out = x
        for conv in self.block_list:
            out = conv(out)
        return out


class Stage(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        num_blocks,
        num_modules,
        multi_scale_output=True,
        name=None,
        align_corners=False,
        padding_same=True,
    ):
        super(Stage, self).__init__()
        self._num_modules = num_modules
        self.stage_list = nn.ModuleList()
        for i in range(num_modules):
            multi_scale = not (i == num_modules - 1 and not multi_scale_output)
            self.stage_list.append(
                HighResolutionModule(
                    input_channels=input_channels,
                    output_channels=output_channels,
                    num_blocks=num_blocks,
                    multi_scale_output=multi_scale,
                    padding_same=padding_same,
                    align_corners=align_corners,
                    name=f"{name}_{i + 1}" if name else None,
                )
            )

    def forward(self, x):
        out = x
        for idx in range(self._num_modules):
            out = self.stage_list[idx](out)
        return out


@BACKBONES.register()
def HRNet_W18(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[18, 36],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[18, 36, 72],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[18, 36, 72, 144],
        **kwargs,
    )
    return model


@BACKBONES.register()
def HRNet_W32(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[32, 64],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[32, 64, 128],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[32, 64, 128, 256],
        **kwargs,
    )
    return model


@BACKBONES.register()
def HRNet_W48(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[48, 96],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[48, 96, 192],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[48, 96, 192, 384],
        **kwargs,
    )
    return model


@BACKBONES.register()
def HRNet_W64(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[64, 128],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[64, 128, 256],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[64, 128, 256, 512],
        **kwargs,
    )
    return model
