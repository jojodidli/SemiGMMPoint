import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation,ProjectionHead
from losses.tree_triplet_loss import TreeTripletLoss
import math
import torch

class get_model(nn.Module):
    def __init__(self, num_classes):
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
        # self.projectionHead = ProjectionHead()

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        # torch.Size([6, 3, 1024]) torch.Size([6, 64, 1024])
        # torch.Size([6, 3, 256]) torch.Size([6, 128, 256])
        # torch.Size([6, 3, 64]) torch.Size([6, 256, 64])
        # torch.Size([6, 3, 16]) torch.Size([6, 512, 16])
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # torch.Size([6, 256, 64])
        # torch.Size([6, 256, 256])
        # torch.Size([6, 128, 1024])
        # torch.Size([6, 128, 4096])
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        # torch.Size([6, 4096, 13]) torch.Size([6, 512, 16])
        # feat: [6,256,512] 其中feat=256
        # return x, self.projectionHead(l4_points)

        # 如果采用l0_point作为feat,此时的输出是[b,128,4096]
        # 要进行归一化，也就是对4096个点的第i个维度进行归一化，需要输入[b,4096,128]
#         l0_points = l0_points.permute(0,2,1)
        l0_points = (torch.nn.functional.normalize(l0_points, p=2, dim=1))

        return x, l0_points


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
        # 'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
        #    'board', 'clutter'
        # 0: ceiling floor wall

        # 1: beam column

        # 2: windows door board

        # 3: table chair sofa 

        # 4: clutter bookcase

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
            # print("2")
            total_loss = F.nll_loss(pred, target, weight=weight)
            target = target.reshape(self.batchsize,-1)
            loss_triplet, class_count = self.treetripletloss(trans_feat, target)
            class_counts = [torch.ones_like(class_count) for _ in range(1)]
            # torch.distributed.all_gather(class_counts, class_count, async_op=False)
            class_counts = torch.cat(class_counts, dim=0)
            # print("loss1",total_loss)

            if 1==torch.nonzero(class_counts, as_tuple=False).size(0): 
            # if torch.distributed.get_world_size()==torch.nonzero(class_counts, as_tuple=False).size(0):
                factor = 1/4*(1+torch.cos(torch.tensor((step.item()-80000)/80000*math.pi))) if step.item()<80000 else 0.5
                total_loss+=factor*loss_triplet
                # print("--------->factor",factor)

            # print("loss2",loss_triplet)
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


if __name__ == '__main__':
    import  torch
    import os
    from torchinfo import summary
    model = get_model(13)
    xyz = torch.rand(6, 9, 4096).to("cuda")
    summary(model,input_data=[xyz])
    a,b = model(xyz)
    print (a.shape,b.shape)
    # # hiera_map = [0,0,1,1,1,2,2,2,3,3,4,5,5,6,6,6,6,6,6]
    # # hiera_index = [[0,2],[2,5],[5,8],[8,10],[10,11],[11,13],[13,19]]
    # # torch.distributed.init_process_group('nccl',init_method='file:///home/.../my_file',world_size=1,rank=0)
    # pred = torch.rand([6*4096,13]).to("cuda")
    # pred = F.log_softmax(pred,dim=1)
    # target = (torch.rand([6*4096]) * 13.).long().to("cuda")
    # feat = torch.rand([6,256,512]).to("cuda")
    # criterion = get_loss(13,6,"hera_embedding").cuda()
    # weight = torch.rand([13]).to("cuda")
    # for i in range(200):
    #     step = (torch.tensor(i)).cuda()
    #     print("step",step)
    #     loss = criterion(step,pred,target,feat,weight)
    #     print(loss)

    criterion2 = get_loss(13,6,"hera_embedding_label").cuda()
    pred = torch.rand([6*4096,13]).to("cuda")
    pred = F.log_softmax(pred,dim=1)
    target = (torch.rand([6*4096]) * 13.).long().to("cuda")
    feat = (torch.rand([6,128,4096])).to("cuda")
    feat = torch.nn.functional.normalize(feat, p=2, dim=1)
    weight = torch.rand([13]).to("cuda")
#     print(feat)

    print("---------------------------")
    step = (torch.tensor(10)).cuda()
    print("step",step)
    loss = criterion2(step,pred,target,feat,weight)
    print(loss)