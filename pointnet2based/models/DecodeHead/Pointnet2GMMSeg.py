import pstats
from cv2 import initUndistortRectifyMap
from einops import rearrange
import torch
import torch.nn as nn

from .GMMHead import GMMSegHead
from .utils import *
import torch.nn.functional as F
import math
from .losses import *

class Pointnet2GMMSegHead(GMMSegHead):

    def __init__(self, 
                in_channels,
                channels,
                num_classes,
                decoder_params,
                dropout_ratio=0.1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=dict(type='ReLU'),
                in_index=-1,
                input_transform='multiple_select',
                loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                ignore_index=255,
                sampler=None,
                align_corners=False,
                interpolate_mode='bilinear'):
        super(Pointnet2GMMSegHead,self).__init__(
                in_channels,
                channels,
                num_classes,
                decoder_params,
                dropout_ratio,
                conv_cfg,
                norm_cfg,
                act_cfg,
                in_index,
                input_transform,
                loss_decode,
                ignore_index,
                sampler,
                align_corners)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(

                nn.Sequential(

                    nn.Conv2d(self.in_channels[i],self.channels, kernel_size=1,stride=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU())
                )

        self.fusion_conv = nn.Sequential(
                    nn.Conv2d(self.channels * num_inputs,self.channels, kernel_size=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU()
            )
            
    def label_transform(self,inputs):
        inputs = rearrange(inputs, "b (h w) -> b 1 h w", h=(int)(math.sqrt(inputs.shape[1]))) 
        return inputs
    def base_feature_transform(self, inputs):


        inputs = self._transform_inputs(inputs)
        for idx in range(len(inputs)):
            x = inputs[idx]
            inputs[idx] = rearrange(x, "b c (h w) -> b c h w", h=(int)(math.sqrt(x.shape[2]))) 

        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        if self.dropout is not None:
            out = self.dropout(out)

        if self.projection is not None:
            out = self.projection(out)
        return out 

    def losses(self, seg_logit, seg_label):

        """Compute segmentation loss."""

        loss = dict()
        seg_logit =rearrange(seg_logit, "b c h w -> b c (h w)")
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss
