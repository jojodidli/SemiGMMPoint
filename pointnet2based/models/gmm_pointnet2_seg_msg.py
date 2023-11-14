import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation,ProjectionHead
from tree_triplet_loss import TreeTripletLoss
import math
import torch

from DecodeHead import Pointnet2GMMSegHead

class get_model(nn.Module):
    def __init__(self, num_classes,class_weight=None):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        self.class_weight = class_weight
        print("class weight",self.class_weight)
        decoder_params=dict(
            embed_dim=64,
            num_components=5,
            gamma=[0.999,0],
            factor_n=1,
            factor_c=1,
            factor_p=1,
            mem_size = 32000,
            max_sample_size=20,
            update_GMM_interval=5,
        )
        in_channels=[128,128,256,256]
        in_index=[0, 1, 2, 3]
        channels=256
        dropout_ratio=0.1
        num_classes=num_classes
        align_corners=False
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,class_weight=self.class_weight)
        self.train_cfg=dict(
            contrast_loss=True,
            contrast_loss_weight=0.01,
            eval = False,
        )

        self.decodeHead = Pointnet2GMMSegHead(
                in_channels=in_channels,
                channels=channels,
                num_classes=num_classes,
                decoder_params=decoder_params,
                dropout_ratio=dropout_ratio,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=dict(type='ReLU'),
                in_index=in_index,
                input_transform='multiple_select',
                loss_decode=loss_decode,
                ignore_index=255,
                sampler=None,
                align_corners=align_corners,
                interpolate_mode='bilinear'
    )

    def forward(self, xyz, gt_seg,eval = True):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)

        l0_points = (torch.nn.functional.normalize(l0_points, p=2, dim=1))
        inputs = [l0_points,l1_points,l2_points,l3_points]
        if eval:
            self.train_cfg['eval'] = eval
            losses, seg_logit,feat = self.decodeHead(inputs,gt_seg,self.train_cfg)
            return losses, seg_logit, feat
        else :
            self.train_cfg['eval'] = eval
            losses,feat = self.decodeHead(inputs,gt_seg,self.train_cfg)
            return losses, feat
        return losses, l0_points


class get_loss(nn.Module):
    def __init__(self,
                 num_classes,
                 batch_size,
                 loss_func = "nll_loss",
                 use_sigmoid=False,
                 loss_weight=1.0
                ):
        super(get_loss, self).__init__()
        self.num_classes = num_classes
        self.hiera_map = [0,0,0,1,1,2,2,3,3,3,4,2,4]
        self.hiera_index = [[0,1,2],[3,4],[5,6,11],[7,8,9],[10,12]]
        self.treetripletloss = TreeTripletLoss(self.num_classes, self.hiera_map, self.hiera_index)

        self.hiera_map2 = [0,1,2,3,4,5,6,7,8,9,10,11,12]
        self.hiera_index2 = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]]
        self.treetripletloss_full = TreeTripletLoss(self.num_classes, self.hiera_map2, self.hiera_index2)

        self.loss_func = loss_func
        self.batchsize = batch_size

    def forward(self, step, pred, target, trans_feat, weight):
        total_loss = torch.tensor(0.)
        factor = 0
        loss_triplet = torch.tensor(0.)
        if(self.loss_func == "nll_loss"):
            total_loss = F.nll_loss(pred, target, weight=weight)
            return total_loss, factor, loss_triplet
        elif(self.loss_func == "hera_embedding"):
            total_loss = F.nll_loss(pred, target, weight=weight)
            target = target.reshape(self.batchsize,-1)
            loss_triplet, class_count = self.treetripletloss(trans_feat, target)
            class_counts = [torch.ones_like(class_count) for _ in range(1)]
            class_counts = torch.cat(class_counts, dim=0)

            if 1==torch.nonzero(class_counts, as_tuple=False).size(0): 
                factor = 1/4*(1+torch.cos(torch.tensor((step.item()-80000)/80000*math.pi))) if step.item()<80000 else 0.5
                total_loss+=factor*loss_triplet

            return total_loss ,factor, loss_triplet
        elif(self.loss_func == "hera_embedding_label"):
            total_loss = F.nll_loss(pred, target, weight=weight)
            target = target.reshape(self.batchsize,-1)
            loss_triplet, class_count = self.treetripletloss_full(trans_feat, target)
            print(class_count)
            class_counts = [torch.ones_like(class_count) for _ in range(1)]
            class_counts = torch.cat(class_counts, dim=0)
            print(total_loss,loss_triplet)
            if 1==torch.nonzero(class_counts, as_tuple=False).size(0): 
                factor = 1/4*(1+torch.cos(torch.tensor((step.item()-80000)/80000*math.pi))) if step.item()<80000 else 0.5
                total_loss+=factor*loss_triplet

            return total_loss, factor, loss_triplet

        return None
