import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import LOSS
import numpy as np
class NCESoftmaxLoss(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, label):
        bsz = x.shape[0]
        x = x.squeeze()
        loss = self.criterion(x, label)
        return loss

@LOSS.register_module()
class GudiePointContrastLoss(nn.Module):
    def __init__(self, npos, T, label_smoothing = 0.1,isNorm = True, weight = None,is_guide=False):
        super(GudiePointContrastLoss, self).__init__()
        self.T = T
        self.npos = npos
        self.isNorm = isNorm
        self.is_guide = is_guide
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,_feat1,_feat2,seg_pred1,seg_pred2,target1,target2):

        ce_loss = self.criterion(seg_pred1,target1)
        assert len(_feat1.shape)==3
        self_loss = 0
        loss = 0
        for index in range(_feat1.shape[0]):
            feat1 = _feat1[index]
            feat2 = _feat2[index]

            feat1 = feat1[:len(feat1)//2]
            feat2 = feat2[:len(feat2)//2]
            
            if self.isNorm:
                feat1 = feat1 / torch.norm(feat1, p=2, dim=1, keepdim=True)
                feat2 = feat2 / torch.norm(feat2, p=2, dim=1, keepdim=True)
            q = feat1
            k = feat2

            if self.npos < q.shape[0]:
                sampled_inds = np.random.choice(q.shape[0], self.npos, replace=False)
                q = q[sampled_inds]
                k = k[sampled_inds]
            npos = q.shape[0] 
            logits = torch.mm(q, k.transpose(1, 0)) # npos by npos 
            labels = torch.arange(npos).cuda().long()# 
            out = torch.div(logits, self.T)
            out = out.squeeze().contiguous()
            if(index ==0 ):
                loss = self.criterion(out,labels)
            else:
                loss += self.criterion(out,labels)
        loss/=_feat1.shape[0]
        # return loss
        for idxx in range(len(target1)):
            feat1 = _feat1[idxx]
            feat2 = _feat2[idxx]
            feat1 = feat1 / torch.norm(feat1, p=2, dim=1, keepdim=True)
            feat2 = feat2 / torch.norm(feat2, p=2, dim=1, keepdim=True)
            feat1 = feat1[:len(feat1)//2]
            feat2 = feat2[:len(feat2)//2]

            
            qs_idx = target1[idxx][:len(target1[idxx])//2].argsort()
            tq1 = target1[idxx][:len(target1[idxx])//2][qs_idx]
            tq2 = target2[idxx][:len(target1[idxx])//2][qs_idx]
            feat1 = feat1[qs_idx]
            feat2 = feat2[qs_idx]
            q_unique, count = tq1.unique(return_counts=True) 
            
            min_iter = torch.min(count).item()//3
            loss_item = 0
            for iter_num in range(min_iter):

                uniform = torch.distributions.Uniform(0, 1).sample([len(count)]).to(target1.device)#【8078】
                off = torch.floor(uniform*count).long()#【8078】
                uniform2 = torch.distributions.Uniform(0, 1).sample([len(count)]).to(target1.device)#【8078】
                off2 = torch.floor(uniform2*count).long()#【8078】
                
                cums = torch.cat([torch.tensor([0], device=count.device), torch.cumsum(count, dim=0)[0:-1]], dim=0) #count的前缀和
                
                _q = feat1[off+cums] 
                _k = feat2[off2+cums].clone().detach()
                
                qt = tq1[off+cums]
                kt = tq2[off2+cums]
                
                assert torch.sum(qt-kt).item() ==0
                
                npos = len(_q)
                logits = torch.mm(_q, _k.transpose(1, 0)) # npos by npos 
                labels = torch.arange(npos).cuda().long()# 
                out = torch.div(logits, self.T)


                out = out.squeeze().contiguous()
                if torch.isnan(out).any():
                    print(_q,_k)
                    print("---------------------------------------")
                    print(out)
                    print("out=Nan")
                    exit(0)
                ii = self.criterion(out,labels)

                if(torch.isnan(ii).any()):
                    print(out)
                    print("---------------------------------------")
                    print("ii is nan")
                    exit(0)
                
                if not torch.isnan(ii).any():
                    loss_item += ii
            loss_item/=min_iter
            self_loss += loss_item
        self_loss/= _feat1.shape[0]
        totle_loss = self_loss + 0.1*loss + 0.02*ce_loss

            
        return totle_loss

    
        



        


