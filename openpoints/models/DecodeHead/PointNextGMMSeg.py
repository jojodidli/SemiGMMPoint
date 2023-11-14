import pstats
from cv2 import initUndistortRectifyMap
from einops import rearrange
import torch
import torch.nn as nn

from .GMMHead import GMMSegHead
# from utils.wapper import resize
from .utils import *
import torch.nn.functional as F
import math
from .losses import *
from ..build import MODELS, build_model_from_cfg


@MODELS.register_module()
class PointNextGMMSeg(GMMSegHead):

    def __init__(self, 
                num_classes,
                in_channels=None,
                channels=128,
                decoder_params=None,
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
                ignore_index=-100,
                sampler=None,
                align_corners=False,
                interpolate_mode='nearest'):
        self.class_weight = None
        print("class weight",self.class_weight)
        decoder_params=dict(
            # * basic setup
            embed_dim=64,
            num_components=5,
            gamma=[0.999,0],
            # * sinkhorn
            factor_n=1,
            factor_c=1,
            factor_p=1,
            # *
            # mem_size=32000,
            mem_size = 36000,
            max_sample_size=20,
            # *
            update_GMM_interval=5,
        )

        in_channels=[32]
        in_index=[0]
        channels=64
        
        dropout_ratio=0.1
        num_classes=num_classes
        align_corners=False
        loss_decode=dict(
            type='CrossEntropyLoss', ignore_index=-100, use_sigmoid=False, loss_weight=1.0,class_weight=self.class_weight)
        self.train_cfg=dict(
            contrast_loss=True,
            contrast_loss_weight=0.01,
            eval = True,
        )
        super(PointNextGMMSeg,self).__init__(
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

                    nn.Conv1d(self.in_channels[i],self.channels, kernel_size=1,stride=1),
                    nn.BatchNorm1d(self.channels),
                    nn.ReLU())
                )

        self.fusion_conv = nn.Sequential(
                    nn.Conv1d(self.channels * num_inputs,self.channels, kernel_size=1),
                    nn.BatchNorm1d(self.channels),
                    nn.ReLU()
            )
        embedding_dim= decoder_params['embed_dim']
        if self.channels != embedding_dim:
            self.projection = nn.Sequential(
                nn.Conv1d(self.channels, embedding_dim, kernel_size=1),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU())
        else: self.projection = None
            
    def label_transform(self,inputs):
        inputs = rearrange(inputs, "b n -> b 1 n")
        return inputs
    def base_feature_transform(self, inputs):


        inputs = self._transform_inputs(inputs)


        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners,D=1))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        if self.dropout is not None:
            out = self.dropout(out)

        if self.projection is not None:
            out = self.projection(out)
        return out 

    def losses(self, seg_logit, seg_label):


        """Compute segmentation loss."""

        loss = dict()
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

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
