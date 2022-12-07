#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, meshgrid

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, strides=[8, 16, 32],
                 in_channels=[256, 512, 1024], act="silu", depthwise=False):
        """
        :param num_classes: 预测类别数
        :param width: 确定网络的宽度系数  通道数系数   0.5
        :param strides: 三个预测特征层的下采样系数 [8, 16, 32]
        :param in_channels: [256, 512, 1024]
        :param act: 激活函数 默认silu
        :param depthwise: 是否使用深度可分离卷积 False
        """
        super().__init__()

        self.n_anchors = 1  # anchor free 每个网格只需要预测1个框
        self.num_classes = num_classes  # 分类数
        self.decode_in_inference = True  # for deploy, set to False

        # 初始化
        self.cls_convs = nn.ModuleList()  # CBL+CBL
        self.reg_convs = nn.ModuleList()  # CBL+CBL
        self.cls_preds = nn.ModuleList()  # Conv
        self.reg_preds = nn.ModuleList()  # Conv
        self.obj_preds = nn.ModuleList()  # Conv
        self.stems = nn.ModuleList()      # BaseConv
        Conv = DWConv if depthwise else BaseConv

        # 遍历三个尺度
        for i in range(len(in_channels)):
            # stem = BaseConv x 3个尺度
            self.stems.append(
                BaseConv(  # 1x1conv
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            # cls_convs = (CBL+CBL) x 3个尺度
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            # reg_convs = (CBL+CBL) x 3个尺度
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            # cls_preds = Conv x 3个尺度
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            # reg_preds = Conv x 3个尺度
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            # obj_preds = Conv x 3个尺度
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False   # 默认False
        # 初始化三个损失函数
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides  # 三个特征层的下采样率 8 16 32
        self.grids = [torch.zeros(1)] * len(in_channels)  # 初始化每个特征层的每个网格的左上角坐标

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        """
        :param xin: {tuple:3} neck输出3个不同尺度的预测特征层
                    0=[bs,128,h/8,w/8]  1=[bs,256,h/16,w/16]  2=[bs,512,h/32,w/32]
        :param labels: [bs,120,cls+xywh]
        :param imgs: [bs,3,w,h]
        :return:
        """
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        # 分别遍历3个层预测特征层  下面以第一层预测进行分析
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)):
            x = self.stems[k](x)   # 1x1 Conv [bs,128,h/8,w/8] -> [bs,128,h/8,w/8]
            cls_x = x              # [bs,128,h/8,w/8]
            reg_x = x              # [bs,128,h/8,w/8]

            cls_feat = cls_conv(cls_x)  # 2xCLB 3x3Conv s=1  [bs,128,h/8,w/8] -> [bs,128,h/8,w/8] -> [bs,128,h/8,w/8]
            cls_output = self.cls_preds[k](cls_feat)  # [bs,128,h/8,w/8] -> [bs,num_classes,h/8,w/8]

            reg_feat = reg_conv(reg_x)  # 2xCLB 3x3Conv s=1  [bs,128,h/8,w/8] -> [bs,128,h/8,w/8] -> [bs,128,h/8,w/8]
            reg_output = self.reg_preds[k](reg_feat)  # [bs,128,h/8,w/8] -> [bs,4(xywh),h/8,w/8]
            obj_output = self.obj_preds[k](reg_feat)  # [bs,128,h/8,w/8] -> [bs,1,h/8,w/8]

            if self.training:
                # [bs,4(xywh),h/8,w/8] [bs,1,h/8,w/8] [bs,num_classes,h/8,w/8] -> [bs,4+1+num_classes,h/8,w/8]
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                # 将当前特征层每个网格的预测输出解码到相对原图上  并得到每个网格的左上角坐标
                # output: 当前特征层的每个网格的解码预测输出 [bs, 80x80, xywh(相对原图)+1+num_classes]
                # grid: 当前特征层每个网格的左上角坐标 [1, 80x80, wh]
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])  # 得到3个特征层每个网格的左上角x坐标  [1,80x80] [1,40x40] [1,20x20]
                y_shifts.append(grid[:, :, 1])  # 得到3个特征层每个网格的左上角y坐标  [1,80x80] [1,40x40] [1,20x20]
                expanded_strides.append(        # 得到当前特征层每个网格的步长  [1,80x80]全是8 [1,40x40]全是16 [1,20x20]全是32
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:   # 默认False
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())
            else:
                # [bs,4(xywh),h/8,w/8] [bs,1,h/8,w/8] [bs,num_classes,h/8,w/8] -> [bs,4+1+num_classes,h/8,w/8]
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)

            outputs.append(output)

        # 【预测阶段】
        # outputs: {list:3}  注意这里得到的4 xywh都是预测的边界框回归参数
        #          0=[bs,4+1+num_classes,h/8,w/8]  1=[bs,num_classes+4+1,h/16,w/16]  2=[bs,4+1+num_classes,h/32,w/32]
        # 【训练阶段】
        # outputs: {list:3}  注意这里得到的4 xywh都是解码后的相对原图的边界框坐标
        # 0=[bs,h/8xw/8,4+1+num_classes] 1=[bs,h/16xw/16,4+1+num_classes] 2=[bs,h/32xw/32,4+1+num_classes]

        if self.training:
            return self.get_losses(imgs, x_shifts, y_shifts, expanded_strides,
                                   labels, torch.cat(outputs, 1), origin_preds, dtype=xin[0].dtype)
        else:
            # {list:3} 0=[h/8,w/8]  1=[h/16,w/16]  2=[h/32,w/32]
            self.hw = [x.shape[-2:] for x in outputs]
            # [bs, n_anchors_all, 4+1+num_classes] = [bs,h/8*w/8 + h/16*w/16 + h/32*w/32, 4+1+num_classes]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            # 解码
            # [bs, n_anchors_all, 4(预测的回归参数)+1+num_classes] -> [bs, n_anchors_all, 4(相对原图的坐标)+1+num_classes]
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        """
        :param output: 网络预测的结果 [bs, xywh(回归参数)+1+num_classes, 80, 80]
        :param k: 第k层预测特征层  0
        :param stride: 当前层stride  8
        :param dtype: 'torch.cuda.HalfTensor'
        :return output: 当前特征层的每个网格的解码预测输出 [bs, 80x80, xywh(相对原图)+1+num_classes]
        :return grid: 当前特征层每个网格的左上角坐标 [1, 80x80, hw]
        """
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]  # 特征层h w
        # 生成当前特征层上每个网格的左上角坐标 self.grids[0]=[1,1,80,80,2(hw)]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        # [bs,xywh(回归参数)+1+num_classes,80,80] -> [bs,1,xywh(回归参数)+1+num_classes,80,80]
        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        # [bs,1,xywh(回归参数)+1+num_classes,80,80] -> [bs,1,80,80,xywh(回归参数)+1+num_classes] -> [bs,1x80x80,xywh(回归参数)+1+num_classes]
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )

        # [1,1,80,80,2(hw)] -> [1, 1x80x80, 2(hw)]
        grid = grid.view(1, -1, 2)

        # 解码
        # 相对原图的xy = (网格左上角坐标 + 预测的xy偏移量) * 当前层stride
        # 相对原图的wh = e^(预测wh回归参数) * 当前层stride
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        """
        :param outputs: [bs, n_anchors_all, 4(预测的回归参数)+1+num_classes]
        :param dtype: 'torch.FloatTensor'
        :return outputs: [bs, n_anchors_all, 4(相对原图的坐标)+1+num_classes]
        """
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)  # 得到每一层的每个网格左上角的坐标
        strides = torch.cat(strides, dim=1).type(dtype)  # 每一层的步长

        # 相对原图的xy = (网格左上角坐标 + 预测的xy偏移量) * 当前层stride
        # 相对原图的wh = e^(预测wh回归参数) * 当前层stride
        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(outputs[..., 2:4]) * strides,
            outputs[..., 4:]
        ], dim=-1)
        return outputs

    def get_losses(self, imgs, x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype):
        """
        :param imgs: 一个batch的图片[bs,3,h,w]
        :param x_shifts: 3个特征图每个网格左上角的x坐标 {list:3} 0=[1,h/8xw/8]  1=[1,h/16xw/16]  2=[1,h/32xw/32]
        :param y_shifts: 3个特征图每个网格左上角的y坐标 {list:3} 0=[1,h/8xw/8]  1=[1,h/16xw/16]  2=[1,h/32xw/32]
        :param expanded_strides: 3个特征图每个网格对应的stride {list:3} 0=[1,h/8xw/8]全是8  1=[1,h/16xw/16]全是16  2=[1,h/32xw/32]全是32
        :param labels: 一个batch的gt [bs,120,class+xywh]  规定每张图片最多有120个目标  不足的部分全部填充为0
        :param outputs: 3个特征图每个网格预测的预测框   注意这里的xywh是相对原图的坐标
                        [bs,h/8xw/8+h/16xw/16+h/32xw/32,xywh+1+num_classes]=[bs,n_anchors_all,xywh+1+num_classes]
        :param origin_preds: []
        :param dtype: torch.float16
        :return:
        """
        bbox_preds = outputs[:, :, :4]  # [bs, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [bs, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [bs, n_anchors_all, num_classes]

        # 计算每张图片有多少个gt框   [bs,]   例如：tensor([5, 5], device='cuda:0')
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)

        # 总的anchor point个数 = 总的网格个数 = total_num_anchors = h/8*w/8 + h/16*w/16 + h/32*w/32
        total_num_anchors = outputs.shape[1]

        x_shifts = torch.cat(x_shifts, 1)  # 3个特征的所有网格的左上角x坐标 [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # 3个特征的所有网格的左上角y坐标 [1, n_anchors_all]

        expanded_strides = torch.cat(expanded_strides, 1)  # 3个特征的所有网格对应的下采样倍率 [1, n_anchors_all]

        if self.use_l1:  # 默认不执行
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        # 遍历每一张图片
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])  # 当前图片的gt个数
            num_gts += num_gt   # 总的gt个数
            if num_gt == 0:  # 默认不执行
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]  # 当前图片所有gt的坐标 [1,num_gt,4(xywh)]
                gt_classes = labels[batch_idx, :num_gt, 0]  # 当前图片所有gt的类别 [bs,num_gt,1]
                bboxes_preds_per_image = bbox_preds[batch_idx]  # 当前图片的所有预测框 [n_anchors_all,4(xywh)]

                # 调用SimOTA正负样本匹配策略
                try:
                    # gt_matched_classes: 每个正样本所匹配到的真实框所属的类别 [num_fg,]
                    # fg_mask: 记录哪些anchor是正样本 哪些是负样本 [total_num_anchors,] True/False
                    # pred_ious_this_matching: 每个正样本与所属的真实框的iou  [num_fg,]
                    # matched_gt_inds: 每个正样本所匹配的真实框idx  [num_fg,]
                    # num_fg: 最终这张图片的正样本个数
                    (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img) = \
                        self.get_assignments(batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image,
                                             gt_classes, bboxes_preds_per_image, expanded_strides, x_shifts,
                                             y_shifts, cls_preds, bbox_preds, obj_preds, labels,imgs)
                except RuntimeError as e:   # 不执行
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise  # RuntimeError might not caused by CUDA OOM

                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()  # 情况显存
                num_fg += num_fg_img  # 当前batch张图片的总正样本数

                # 独热编码 每个正样本所匹配到的真实框所属的类别 [num_fg,] -> [num_fg, num_classes]
                # 得到当前图片的gt class  [num_fg, num_classes]
                cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) * pred_ious_this_matching.unsqueeze(-1)
                # 得到当前图片的gt obj  [8400, 1]
                obj_target = fg_mask.unsqueeze(-1)
                # 得到当前图片的gt box [num_gt, xywh]
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        # 假设batch张图片所有的正样本个数 = P
        # batch张图片的所有正样本对应的gt class  独热编码   {list:bs} -> [P, 80]
        cls_targets = torch.cat(cls_targets, 0)
        # batch张图片的所有正样本对应的gt box  {list:bs} -> [P, 4]
        reg_targets = torch.cat(reg_targets, 0)
        # batch张图片的所有正样本对应的gt obj  {list:bs} -> [bsx8400, 1]
        obj_targets = torch.cat(obj_targets, 0)
        # [bsx8400]  记录batch张图片的所有anchor point哪些anchor是正样本 哪些是负样本  True/False
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        # 分别计算3个loss
        num_fg = max(num_fg, 1)   # batch张图片所有的正样本个数
        # 回归损失: iou loss 正样本
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
        # 置信度损失: 交叉熵损失 正样本 + 负样本
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
        # 分类损失: 交叉熵损失 正样本
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num_fg
        if self.use_l1:
            loss_l1 = (self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg
        else:
            loss_l1 = 0.0

        # 合并总loss
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (loss, reg_weight * loss_iou, loss_obj, loss_cls, loss_l1, num_fg / max(num_gts, 1))

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(self, batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes,
                        bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts, cls_preds,
                        bbox_preds, obj_preds, labels, imgs, mode="gpu"):
        """正负样本匹配
        :param batch_idx: 第几张图片
        :param num_gt: 当前图片的gt个数
        :param total_num_anchors: 当前图片总的anchor point个数  640x640 -> 80x80+40x40+20x20 = 8400
        :param gt_bboxes_per_image: [num_gt, 4(xywh相对原图)] 当前图片的gt box
        :param gt_classes: [num_gt,] 当前图片的gt box所属类别
        :param bboxes_preds_per_image: [total_num_anchors, xywh(相对原图)] 当前图片的每个anchor point相对原图的预测box坐标
        :param expanded_strides: [1, total_num_anchors]  当前图片每个anchor point的下采样倍率
        :param x_shifts: [1, total_num_anchors] 当前图片每个anchor point的网格左上角x坐标
        :param y_shifts: [1, total_num_anchors] 当前图片每个anchor point的网格左上角y坐标
        :param cls_preds: [bs, total_num_anchors, num_classes] bs张图片每个anchor point的预测类别
        :param bbox_preds: [bs, total_num_anchors, 4(xywh相对原图)] bs张图片每个anchor point相对原图的预测box坐标
        :param obj_preds: [bs, total_num_anchors, 1] bs张图片每个anchor point相对原图的预测置信度
        :param labels: [bs, 200, class+xywh]  batch张图片的原始gt信息  每张图片最多200个gt  不足的全是0
        :param imgs: [bs, 3, 640, 640] 输入batch张图片
        :param mode: 'gpu'
        :return gt_matched_classes: 每个正样本所匹配到的真实框所属的类别 [num_fg,]
        :return fg_mask: 记录哪些anchor是正样本 哪些是负样本 [total_num_anchors,] True/False
        :return pred_ious_this_matching: 每个正样本与所属的真实框的iou  [num_fg,]
        :return matched_gt_inds: 每个正样本所匹配的真实框idx  [num_fg,]
        :return num_fg: 最终这张图片的正样本个数
        """
        if mode == "cpu":   # 默认不执行
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        # 1、确定正样本候选区域（使用中心先验）
        # fg_mask: [total_num_anchors] gt内部和中心区域内部的所有anchor point都是候选框  所以是两者的并集
        #          True/False   假设所有True的个数为num_candidate
        # is_in_boxes_and_center: [num_gt, num_candidate]  对应这张图像每个gt的候选框anchor point True/False
        #                         而且这些候选框anchor point是既在gt框内部也在fixed center area区域内的
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts,
                                                                 y_shifts, total_num_anchors, num_gt)

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]  # 得到当前图片所有候选框的预测box [num_candidate, xywh(相对原图)]
        cls_preds_ = cls_preds[batch_idx][fg_mask]  # 得到当前图片所有候选框的预测cls [num_candidate, num_classes]
        obj_preds_ = obj_preds[batch_idx][fg_mask]  # 得到当前图片所有候选框的预测obj [num_candidate, 1]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]  # 候选框个数

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        # 2、计算每个候选框anchor point和每个gt的iou矩阵
        # [num_gt, 4(xywh相对原图)] [num_candidate, 4(xywh相对原图)] -> [num_gt, num_candidate]
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        # 3、计算每个候选框和每个gt的cost矩阵
        # gt cls转为独热编码  方便后面计算cls loss
        # [num_gt] -> [num_gt, num_classes] -> [num_gt, 1, num_classes] -> [num_gt, num_candidate, num_classes]
        gt_cls_per_image = (F.one_hot(gt_classes.to(torch.int64), self.num_classes).float()
                            .unsqueeze(1).repeat(1, num_in_boxes_anchor, 1))
        # 计算每个候选框和每个gt的iou loss = -log(iou)  为什么不是1-iou?
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        # 计算每个候选框和每个gt的分类损失pair_wise_cls_loss
        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                          * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_())
            pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
        del cls_preds_

        # 计算每个候选框和每个gt的cost矩阵  [num_gt, num_candidate]
        # 其中cost = cls loss + 3 * iou loss + 100000.0 * (~is_in_boxes_and_center)
        # is_in_boxes_and_center表示gt box和fixed center area交集的区域  取反就是并集-交集的区域
        # 给这些区域的cost取一个非常大的数字 那么在后续的dynamic_k_matching根据最小化cost原则
        # 我们会优先选取这些交集的区域  如果交集区域还不够才回去选取并集-交集的区域
        cost = (pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center))

        # 4、使用iou矩阵，确定每个gt的dynamic_k
        # num_fg: 最终的正样本个数
        # gt_matched_classes: 每个正样本所匹配到的真实框所属的类别 [num_fg,]
        # pred_ious_this_matching: 每个正样本与所属的真实框的iou  [num_fg,]
        # matched_gt_inds: 每个正样本所匹配的真实框idx  [num_fg,]
        (num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds) = \
            self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg)

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt):
        """确定正样本候选区域
        :param gt_bboxes_per_image: [num_gt, 4(xywh相对原图的)] 当前图片的gt box
        :param expanded_strides: [1, total_num_anchors]  当前图片每个anchor point的下采样倍率
        :param x_shifts: [1, total_num_anchors] 当前图片每个anchor point的网格左上角x坐标
        :param y_shifts: [1, total_num_anchors] 当前图片每个anchor point的网格左上角y坐标
        :param total_num_anchors: 当前图片总的anchor point个数  640x640 -> 80x80+40x40+20x20 = 8400
        :param num_gt: 当前图片的gt个数
        :return is_in_boxes_anchor: [total_num_anchors] gt内部和中心区域内部的所有anchor point都是候选框  所以是两者的并集
                                    True/False   假设所有True的个数为num_candidate
        :return is_in_boxes_and_center: [num_gt, num_candidate]  对应这张图像每个gt的候选框anchor point True/False
                                        而且这些候选框anchor point是既在gt框内部也在fixed center area区域内的
        """
        # 一、计算哪些网格的中心点是在gt内部的
        # 计算每个网格的中心点坐标
        # [total_num_anchors,] 当前图片的3个特征图中每个grid cell的缩放比
        expanded_strides_per_image = expanded_strides[0]
        # [total_num_anchors,] 当前图片3个特征图中每个grid cell左上角在原图上的x坐标
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        # [total_num_anchors,] 当前图片3个特征图中每个grid cell左上角在原图上的y坐标
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        # 得到每个网格中心点的x坐标（相对原图） [total_num_anchors,] -> [1, total_num_anchors] -> [num_gt, total_num_anchors]
        x_centers_per_image = ((x_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1))
        # 得到每个网格中心点的y坐标（相对原图） [total_num_anchors,] -> [1, total_num_anchors] -> [num_gt, total_num_anchors]
        y_centers_per_image = ((y_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1))

        # 计算所有gt框相对原图的左上角和右下角坐标  gt: [num_gt, 4(xywh)]  xy为中心点坐标  wh为宽高
        # 计算每个gt左上角的x坐标  x - 0.5 * w      [num_gt, ] -> [num_gt, 1] -> [num_gt, total_num_anchors]
        gt_bboxes_per_image_l = ((gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors))
        # 计算每个gt右下角的x坐标  x + 0.5 * w      [num_gt, ] -> [num_gt, 1] -> [num_gt, total_num_anchors]
        gt_bboxes_per_image_r = ((gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors))
        # 计算每个gt左上角的y坐标  y - 0.5 * h      [num_gt, ] -> [num_gt, 1] -> [num_gt, total_num_anchors]
        gt_bboxes_per_image_t = ((gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors))
        # 计算每个gt右下角的y坐标  y + 0.5 * h      [num_gt, ] -> [num_gt, 1] -> [num_gt, total_num_anchors]
        gt_bboxes_per_image_b = ((gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors))

        # 计算哪些网格的中心点是在gt内部的
        # 每个网格中心点x坐标 - 每个gt左上角的x坐标
        b_l = x_centers_per_image - gt_bboxes_per_image_l  # [num_gt, total_num_anchors]
        # 每个gt右下角的x坐标 - 每个网格中心点x坐标
        b_r = gt_bboxes_per_image_r - x_centers_per_image  # [num_gt, total_num_anchors]
        # 每个网格中心点的y坐标 - 每个gt左上角的y坐标
        b_t = y_centers_per_image - gt_bboxes_per_image_t  # [num_gt, total_num_anchors]
        # 每个gt右下角的y坐标 - 每个网格中心点的y坐标
        b_b = gt_bboxes_per_image_b - y_centers_per_image  # [num_gt, total_num_anchors]
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2) # 4x[num_gt, total_num_anchors] -> [num_gt, total_num_anchors, 4]
        # b_l, b_t, b_r, b_b中最小的一个>0.0 则为True  也就是说要保证b_l, b_t, b_r, b_b四个都大于0 此时说明这个网格中心点位于这个gt的内部(可以画个图理解下)
        # [num_gt, total_num_anchors]  True表示当前这个网格是落在这个gt内部的
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        # [total_num_anchors]  某个网格只要落在一个gt内部就是True   否则False
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        # 二、计算哪些网格是在fixed center area区域内  计算步骤和一是一样的 就不赘述了
        # fixed center area  中心区域大小是 (5xstride) x (5xstride)  中心点是每个gt的中心点
        center_radius = 2.5
        # 计算所有中心区域相对原图的左上角和右下角坐标  [num_gt, total_num_anchors]
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) \
                                - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) \
                                + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) \
                                - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) \
                                + center_radius * expanded_strides_per_image.unsqueeze(0)

        # 计算哪些网格的中心点是在fixed center area区域内的
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        # [total_num_anchors]  某个网格只要落在一个中心区域内部就是True   否则False
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # 三、得到最终的所有的c
        # is_in_boxes_anchor: [total_num_anchors] gt内部和中心区域内部的所有anchor point都是候选框  所以是两者的并集
        #                     True/False   假设所有True的个数为num_candidate
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        # is_in_boxes_and_center: [num_gt, num_candidate]  对应这张图像每个gt的候选框anchor point True/False
        # &: 表示这些候选框anchor point是既在gt框内部也在fixed center area区域内的
        is_in_boxes_and_center = (is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor])

        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        """确定每个gt的dynamic_k
        正样本筛选过程：8400 -> num_candidate -> num_fg
        :param cost: 每个候选框和每个gt的cost矩阵  [num_gt, num_candidate]
        :param pair_wise_ious: 每个候选框和每个gt的iou矩阵 [num_gt, num_candidate]
        :param gt_classes: 当前图片的gt box所属类别 [num_gt,]
        :param num_gt: 当前图片的gt个数
        :param fg_mask: [total_num_anchors,] gt内部和中心区域内部的所有anchor point都是候选框  所以是两者的并集
                        True/False   假设所有True的个数为num_candidate
        :return num_fg: 最终的正样本个数
        :return gt_matched_classes: 每个正样本所匹配到的真实框所属的类别 [num_fg,]
        :return pred_ious_this_matching: 每个正样本与所属的真实框的iou  [num_fg,]
        :return matched_gt_inds: 每个正样本所匹配的真实框idx  [num_fg,]
        """
        # 初始化匹配矩阵 [num_gt, num_candidate]
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious

        # 每个gt选取前topk个iou
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        # [num_gt, num_candidate] -> [num_gt, 10]
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        # 再把每个gt的topk个iou相加求出每个gt的正样本数量(>=1)  [num_gt,]
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        # {list:num_gt}  [5, 6, 4, 7, 5, 7, 4, 4, 7, 6, 8]  对应每个gt的正样本数量
        dynamic_ks = dynamic_ks.tolist()
        # 遍历每个gt, 选取前dynamic_ks个最小的cost对应的anchor point作为最终的正样本
        for gt_idx in range(num_gt):
            # pos_idx: 正样本对应的idx
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
            # 把匹配矩阵的gt和anchor point对应的idx置为1 意为这个anchor point是这个gt的正样本
            matching_matrix[gt_idx][pos_idx] = 1
        del topk_ious, dynamic_ks, pos_idx

        # 消除重复匹配: 如果有1个anchor point是多个gt的正样本，那么还是最小化原则，它是cost最小的那个gt的正样本，其他gt的负样本
        # 计算每个候选anchor point匹配的gt个数  [num_candidate,]
        anchor_matching_gt = matching_matrix.sum(0)
        # 如果大于1 说明有1个anchor分配给了多个gt  那么要重新分配这个anchor：把这个anchor分配给cost小的那个gt
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)  # 取cost小的位置idx
            matching_matrix[:, anchor_matching_gt > 1] *= 0            # 重复匹配的区域（大于1）全为0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1   # cost小的改为1

        # fg_mask_inboxes: [num_candidate] True/False  最终的正样本区域为True  负样本为False
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        # 最终的正样本总个数
        num_fg = fg_mask_inboxes.sum().item()

        # fg_mask: [total_num_anchors]  True/False  最终的正样本区域为True  负样本为False
        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        # 每个正样本所匹配的真实框idx  [num_fg,]  注意每个真实框可能会有多个正样本，但是每个正样本只能是一个真实框的正样本
        # [num_gt, num_candidate] -> [num_gt, num_fg] -> [num_fg,]
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        # 每个正样本所匹配到的真实框所属的类别 [num_fg,]
        gt_matched_classes = gt_classes[matched_gt_inds]

        # 每个正样本与所属的真实框的iou  [num_fg,]
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
