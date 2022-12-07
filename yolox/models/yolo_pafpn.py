#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"),
                 in_channels=[256, 512, 1024], depthwise=False, act="silu"):
        """
        :param depth: 确定网络的深度系数  卷积的个数  0.33
        :param width: 确定网络的宽度系数  通道数     0.5
        :param in_features: backbone输出的三个特征名
        :param in_channels: backbone输出 并 传入head三个特征的channel
        :param depthwise: 是否使用深度可分离卷积  默认False
        :param act: 激活函数 默认silu
        """
        super().__init__()  # 继承父类的init方法
        # 创建backbone
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features  # ("dark3", "dark4", "dark5")
        self.in_channels = in_channels  # [256, 512, 1024]
        Conv = DWConv if depthwise else BaseConv

        # 上采样1
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(   # 512 -> 256
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        # upsample + concat -> 512
        self.C3_p4 = CSPLayer(    # 512 -> 256
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # 上采样2
        self.reduce_conv1 = BaseConv(   # 256 -> 128
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        # upsample + concat -> 256
        self.C3_p3 = CSPLayer(     # 256 -> 128
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # 下采样1  bottom-up conv
        self.bu_conv2 = Conv(  # 128 -> 128  3x3conv s=2
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        # concat 128 -> 256
        self.C3_n3 = CSPLayer(   # 256 -> 256
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # 上采样2  bottom-up conv
        self.bu_conv1 = Conv(  # 256 -> 256   3x3conv s=2
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        # concat 256 -> 512
        self.C3_n4 = CSPLayer(   # 512 -> 512
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        :param input: 一个batch的输入图片 [bs,3,h,w]
        :return outputs: {tuple:3}  neck输出3个不同尺度的预测特征层
                         0=[bs,128,h/8,w/8]  1=[bs,256,h/16,w/16]  2=[bs,512,h/32,w/32]
        """
        # backbone  {dict:3}
        # 'dark3'=[bs,128,h/8,w/8]  'dark4'=[bs,256,h/16,w/16]  'dark5'=[bs,512,h/32,w/32]
        out_features = self.backbone(input)
        # list:3  [bs,128,h/8,w/8]  [bs,256,h/16,w/16]  [bs,512,h/32,w/32]
        features = [out_features[f] for f in self.in_features]
        # x0=[bs,512,h/32,w/32]   x1=[bs,256,h/16,w/16]  x2=[bs,128,h/8,w/8]
        [x2, x1, x0] = features

        # 上采样1
        # [bs,512,h/32,w/32] -> [bs,256,h/32,w/32]
        fpn_out0 = self.lateral_conv0(x0)
        # [bs,256,h/32,w/32] -> [bs,256,h/16,w/16]
        f_out0 = self.upsample(fpn_out0)
        # [bs,256,h/16,w/16] cat [bs,256,h/16,w/16] -> [bs,512,h/16,w/16]
        f_out0 = torch.cat([f_out0, x1], 1)
        # [bs,512,h/16,w/16] -> [bs,256,h/16,w/16]
        f_out0 = self.C3_p4(f_out0)

        # 上采样2
        # [bs,256,h/16,w/16] -> [bs,128,h/16,w/16]
        fpn_out1 = self.reduce_conv1(f_out0)
        # [bs,128,h/16,w/16] -> [bs,128,h/8,w/8]
        f_out1 = self.upsample(fpn_out1)
        # [bs,128,h/8,w/8] cat [bs,128,h/8,w/8] -> [bs,256,h/8,w/8]
        f_out1 = torch.cat([f_out1, x2], 1)
        # [bs,256,h/8,w/8] -> [bs,128,h/8,w/8]
        pan_out2 = self.C3_p3(f_out1)

        # 下采样1
        # [bs,128,h/8,w/8] -> [bs,128,h/16,w/16]
        p_out1 = self.bu_conv2(pan_out2)
        # [bs,128,h/16,w/16] cat [bs,128,h/16,w/16] -> [bs,256,h/16,w/16]
        p_out1 = torch.cat([p_out1, fpn_out1], 1)
        # [bs,256,h/16,w/16] -> [bs,256,h/16,w/16]
        pan_out1 = self.C3_n3(p_out1)

        # 下采样2
        # [bs,256,h/16,w/16] -> [bs,256,h/32,w/32]
        p_out0 = self.bu_conv1(pan_out1)
        # [bs,256,h/32,w/32] cat [bs,256,h/32,w/32] -> [bs,512,h/32,w/32]
        p_out0 = torch.cat([p_out0, fpn_out0], 1)
        # [bs,512,h/32,w/32] -> [bs,512,h/32,w/32]
        pan_out0 = self.C3_n4(p_out0)

        outputs = (pan_out2, pan_out1, pan_out0)

        # {tuple:3}  neck输出3个不同尺度的预测特征层
        # 0=[bs,128,h/8,w/8]  1=[bs,256,h/16,w/16]  2=[bs,512,h/32,w/32]
        return outputs
