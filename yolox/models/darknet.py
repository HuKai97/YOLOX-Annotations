#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from torch import nn

from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck


class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output channels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"), depthwise=False, act="silu"):
        """
        :param dep_mul: 确定网络的深度  卷积的个数  0.33
        :param wid_mul: 确定网络的宽度  通道数     0.5
        :param out_features: backbone输出的三个特征名
        :param depthwise: 是否使用深度可分离卷积  默认False
        :param act: 激活函数 默认silu
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features  # ("dark3", "dark4", "dark5")
        Conv = DWConv if depthwise else BaseConv  # BaseConv = nn.Conv2d + bn + silu

        base_channels = int(wid_mul * 64)          # 32  stem输出的特征channel数
        base_depth = max(round(dep_mul * 3), 1)    # 1   bottleneck卷积个数

        # stem  [bs,3,w,h] -> [bs,32,w/2,h/2]
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2 = Conv + CSPLayer
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),  # [bs,32,w/2,h/2] -> [bs,64,w/4,h/4]
            CSPLayer(                                               # [bs,64,w/4,h/4] -> [bs,64,w/4,h/4]
                base_channels * 2,
                base_channels * 2,
                n=base_depth,            # 1个bottleneck
                depthwise=depthwise,     # False
                act=act,                 # silu
            ),
        )

        # dark3 = Conv + 3 * CSPLayer
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),  # [bs,64,w/4,h/4] -> [bs,128,w/8,h/8]
            CSPLayer(                                                   # [bs,128,w/8,h/8] -> [bs,128,w/8,h/8]
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,         # 3个bottleneck
                depthwise=depthwise,      # False
                act=act,                  # silu
            ),
        )

        # dark4 = Conv + 3 * CSPLayer
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),  # [bs,128,w/8,h/8] -> [bs,256,w/16,h/16]
            CSPLayer(                                                   # [bs,256,w/16,h/16] -> [bs,256,w/16,h/16]
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,      # 3个bottleneck
                depthwise=depthwise,   # False
                act=act,               # silu
            ),
        )

        # dark5 Conv + SPPBottleneck + CSPLayer
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),             # [bs,256,w/16,h/16] -> [bs,512,w/32,h/32]
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),  # [bs,512,w/32,h/32] -> [bs,512,w/32,h/32]
            CSPLayer(                                                               # [bs,512,w/32,h/32] -> [bs,512,w/32,h/32]
                base_channels * 16,
                base_channels * 16,
                n=base_depth,         # 1个bottleneck
                shortcut=False,       # 没有shortcut
                depthwise=depthwise,  # False
                act=act,              # silu
            ),
        )

    def forward(self, x):
        # x: [bs,3,w,h]
        outputs = {}
        # [bs,3,w,h] -> [bs,32,w/2,h/2]
        x = self.stem(x)
        outputs["stem"] = x
        # [bs,32,w/2,h/2] -> [bs,64,w/4,h/4]
        x = self.dark2(x)
        outputs["dark2"] = x
        # [bs,64,w/4,h/4] -> [bs,128,w/8,h/8]
        x = self.dark3(x)
        outputs["dark3"] = x
        # [bs,128,w/8,h/8] -> [bs,256,w/16,h/16]
        x = self.dark4(x)
        outputs["dark4"] = x
        # [bs,256,w/16,h/16] -> [bs,512,w/32,h/32]
        x = self.dark5(x)
        outputs["dark5"] = x
        # 输出：dark2=[bs,128,w/8,h/8]  dark3=[bs,256,w/16,h/16]  dark4=[bs,512,w/32,h/32]
        return {k: v for k, v in outputs.items() if k in self.out_features}
