

from abc import  abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import * 
from .utils import * 

from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

class GMMSegHead(nn.Module):

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
                input_transform=None,
                loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight = None),
                ignore_index=-100,
                sampler=None,
                align_corners=False,
                
                 
    ):
        super(GMMSegHead, self).__init__()
        self.sampler = sampler
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.input_transform = input_transform
        self.IGNIRE_INDEX = -100

        if isinstance(loss_decode, dict):
            self.loss_decode = self.build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(self.build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.ignore_index = ignore_index
        self.align_corners = align_corners


        decoder_params = decoder_params
        embedding_dim = decoder_params['embed_dim']

        if self.channels != embedding_dim:
            self.projection = nn.Sequential(
                nn.Conv2d(self.channels, embedding_dim, kernel_size=1),
                nn.BatchNorm2d(embedding_dim),
                nn.ReLU())
        else: self.projection = None
        self.embedding_dim = embedding_dim
        self.num_components = decoder_params['num_components']
        self.update_GMM_interval = decoder_params['update_GMM_interval']

        gamma = decoder_params['gamma']
        self.gamma_mean = gamma if isinstance(decoder_params['gamma'],(float,int)) else gamma[0]
        self.gamma_cov = gamma if isinstance(decoder_params['gamma'],(float,int)) else gamma[1]
        self.factors = [decoder_params['factor_n'], decoder_params['factor_c'], decoder_params['factor_p']]

        self.K = decoder_params['mem_size']
        self.Ks = torch.tensor([self.K for _c in range(self.num_classes*self.num_components)], dtype=torch.long)
        
        self.max_sample_size = decoder_params['max_sample_size']
        self.register_buffer("queue", torch.randn(self.num_classes*self.num_components, embedding_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=-2)
        self.register_buffer("queue_ptr", torch.zeros(self.num_classes*self.num_components, dtype=torch.long))

        self.apply(init_weights)

        self.means = nn.Parameter(torch.zeros(self.num_classes, self.num_components, embedding_dim), requires_grad=False)
        trunc_normal_(self.means, std=0.02)
        self.num_prob_n = self.num_components
        self.diagonal = nn.Parameter(torch.ones(self.num_classes,self.num_components,self.embedding_dim), requires_grad=False)
        self.eye_matrix = nn.Parameter(torch.ones(embedding_dim), requires_grad=False)
        self.feat_norm = nn.LayerNorm(embedding_dim) 
        self.mask_norm = nn.LayerNorm(self.num_classes) 
        
        self.iteration_counter = nn.Parameter(torch.zeros(1), requires_grad=False)

    def GetConvModule(self,_in_channels,_out_channels, _kernel_size, _IsPooling= False):
        return nn.Sequential(
            nn.Conv2d(_in_channels, _out_channels, kernel_size=_kernel_size),
            nn.BatchNorm2d(_out_channels),
            nn.ReLU(),
            )
    @abstractmethod
    def base_feature_transform(self, inputs):
        pass
    @abstractmethod
    def label_transform(self, inputs):
        pass


    def forward_help(self, base_feature, gt_semantic_seg=None, train_cfg=None, test_cfg=None):


        _c = rearrange(base_feature, 'b c n -> (b n) c') #[B,64,N]
        _c = self.feat_norm(_c) # * n, d [65536,64]
        _c = l2_normalize(_c)

        self.means.data.copy_(l2_normalize(self.means))
        #
        _log_prob = self.compute_log_prob(_c) # [65536,95]
        final_probs = _log_prob.contiguous().view(-1, self.num_classes, self.num_prob_n)#[65536, 19,5]

        _m_prob = torch.amax(final_probs, dim=-1) #[65536,19]

        out_seg = self.mask_norm(_m_prob)#[65536,19]
        out_seg = rearrange(out_seg, "(b n) k -> b k n", b=base_feature.shape[0], n=base_feature.shape[2]) #[1,19,256,256]
        
        if train_cfg['eval'] is False and gt_semantic_seg is not None: 
            gt_semantic_seg = self.label_transform(gt_semantic_seg) # [B，1,N]
            gt_seg_full = resize(gt_semantic_seg.float(), size=base_feature.size()[2:], mode='nearest',D=1)
            gt_seg = gt_seg_full.view(-1)

            contrast_logits, contrast_target, qs = self.online_contrast(gt_seg, final_probs, _c, out_seg)

            with torch.no_grad():
                # * update memory
                _c_mem = concat_all_gather_wo_grad(_c)
                _gt_seg_mem = concat_all_gather_wo_grad(gt_seg)
                _qs = concat_all_gather_wo_grad(qs)


                unique_c_list = _gt_seg_mem.unique().int()
                for k in unique_c_list:
                    if k == self.IGNIRE_INDEX: continue
                    self._dequeue_and_enqueue_k(k.item(), _c_mem, _qs.bool(), (_gt_seg_mem == k.item()))

                # * EM
                if self.iteration_counter % self.update_GMM_interval == 0:
                    self.update_GMM(unique_c_list)

            return out_seg, contrast_logits, contrast_target

        return out_seg
    

    def forward(self, inputs, gt_semantic_seg, train_cfg):
        base_feature = self.base_feature_transform(inputs) #[1,64,N]
        selfAuto = True
        if selfAuto :
            contrast_logits, contrast_target = None, None
            seg_logits = None
            if train_cfg['eval'] is not True:
                seg_logits, contrast_logits, contrast_target = self.forward_help(base_feature, gt_semantic_seg=gt_semantic_seg, train_cfg=train_cfg)
            else:
                seg_logits = self.forward_help(base_feature, gt_semantic_seg=gt_semantic_seg, train_cfg=train_cfg)
            
            losses = {}
            if train_cfg['eval'] is not True:
                losses = self.losses(seg_logits, gt_semantic_seg)
                
            if train_cfg['contrast_loss'] is True and contrast_logits is not None:
                loss_contrast = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=self.IGNIRE_INDEX)
                losses['loss_contrast'] = loss_contrast * train_cfg['contrast_loss_weight']
            for k in range(2*self.num_classes):
                losses['i_d_loss'] =self.cal_id_loss()
            self.iteration_counter += 1
        else:
            losses = 0
            seg_logits = 0

        return losses, seg_logits,base_feature


    def compute_log_prob(self, _fea):
        covariances = self.diagonal.detach_() # * c,p,d,d

        _prob_n = []
        _n_group = _fea.shape[0] // self.factors[0]
        _c_group = self.num_classes // self.factors[1]
        for _c in range(0,self.num_classes,_c_group):
            _prob_c = []
            _c_means = self.means[_c:_c+_c_group]
            _c_covariances = covariances[_c:_c+_c_group]

            _c_gauss = MultivariateNormalDiag(_c_means.view(-1, self.embedding_dim), scale_diag=_c_covariances.view(-1,self.embedding_dim)) # * c*p multivariate gaussian
            for _n in range(0,_fea.shape[0],_n_group):
                _prob_c.append(_c_gauss.log_prob(_fea[_n:_n+_n_group,None,...]))
            _c_probs = torch.cat(_prob_c, dim=0) # n, cp
            _c_probs = _c_probs.contiguous().view(_c_probs.shape[0], -1, self.num_prob_n)
            _prob_n.append(_c_probs)
        probs = torch.cat(_prob_n, dim=1)

        return probs.contiguous()

        
    @torch.no_grad()
    def _dequeue_and_enqueue_k(self, _c, _c_embs, _c_cluster_q, _c_mask):

        if _c_mask is None: _c_mask = torch.ones(_c_embs.shape[0]).detach_()

        _k_max_sample_size = self.max_sample_size
        _embs = _c_embs[_c_mask>0]
        _cluster = _c_cluster_q[_c_mask>0]
        for q_index in range(self.num_components):
            _q_ptr = _c*self.num_components+q_index
            ptr = int(self.queue_ptr[_q_ptr])
            
            if torch.sum(_cluster[:, q_index]) == 0: continue
            assert _cluster[:, q_index].shape[0] == _embs.shape[0]
            _q_embs = _embs[_cluster[:, q_index]]

            _q_sample_size = _q_embs.shape[0]
            assert _q_sample_size == torch.sum(_cluster[:, q_index])

            if self.max_sample_size != -1 and _q_sample_size > _k_max_sample_size:
                _rnd_sample = rnd_sample(_q_sample_size, _k_max_sample_size, _uniform=True, _device=_c_embs.device)
                _q_embs = _q_embs[_rnd_sample, ...]
                _q_sample_size = _k_max_sample_size

            # replace the embs at ptr (dequeue and enqueue)
            if ptr + _q_sample_size >= self.Ks[_q_ptr]:
                _fir = self.Ks[_q_ptr] - ptr
                _sec = _q_sample_size - self.Ks[_q_ptr] + ptr
                self.queue[_q_ptr, :, ptr:self.Ks[_q_ptr]] = _q_embs[:_fir].T
                self.queue[_q_ptr, :, :_sec] = _q_embs[_fir:].T
            else:
                self.queue[_q_ptr, :, ptr:ptr + _q_sample_size] = _q_embs.T
            
            ptr = (ptr + _q_sample_size) % self.Ks[_q_ptr].item()# move pointer
            self.queue_ptr[_q_ptr] = ptr

    
    @torch.no_grad()
    def update_GMM(self, unique_c_list):
        components = self.means.data.clone() 
        covs = self.diagonal.data.clone()

        for _c in unique_c_list:
            if _c == self.IGNIRE_INDEX: continue
            _c = _c if isinstance(_c, int) else _c.item()

            for _p in range(self.num_components):
                _p_ptr = _c*self.num_components + _p
                _mem_fea_q = self.queue[_p_ptr,:,:self.Ks[_c]].transpose(-1,-2) # n,d

                f = l2_normalize(torch.sum(_mem_fea_q, dim=0)) # d,

                new_value = momentum_update(old_value=components[_c, _p, ...], new_value=f, momentum=self.gamma_mean, debug=False)
                components[_c, _p, ...] = new_value

                _shift_fea = _mem_fea_q - f[None, ...] # * n, d

                _cov = shifted_var(_shift_fea, rowvar=False)
                _cov = _cov + 1e-2 * self.eye_matrix
                _cov = _cov.sqrt()

                new_covariance = momentum_update(old_value=covs[_c, _p, ...], new_value=_cov, momentum=self.gamma_cov, debug=False)
                covs[_c, _p, ...] = new_covariance
        
        self.means = nn.Parameter(components, requires_grad=False)
        self.diagonal = nn.Parameter(covs, requires_grad=False)
    def _transform_inputs(self, inputs):

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs


    def online_contrast(self, gt_seg, simi_logits, _c, out_seg):

        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))

        # compute logits
        contrast_logits = simi_logits.flatten(1) # * n, c*p
        contrast_target = gt_seg.clone().float()


        return_qs = torch.zeros(size=(simi_logits.shape[0], self.num_components), device=gt_seg.device)
        for k in gt_seg.unique().long():
            if k == self.IGNIRE_INDEX: continue
            init_q = simi_logits[:, k, :]
            init_q = init_q[gt_seg == k, ...] # n,p
            init_q = init_q[:,:self.num_components]
            init_q = init_q / torch.abs(init_q).max()

            q, indexs = distributed_sinkhorn_wograd(init_q)
            try:
                assert torch.isnan(q).int().sum() <= 0
            except:
                q[torch.isnan(q)] = 0
                indexs[torch.isnan(q).int().sum(dim=1)>0] = 255 - (self.num_prob_n * k)

            m_k = mask[gt_seg == k]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_components)
            q = q * m_k_tile  # n x self.num_prob_n

            contrast_target[gt_seg == k] = indexs.float() + (self.num_prob_n * k)

            return_qs[gt_seg == k] = q

        return contrast_logits, contrast_target, return_qs


    def build_loss(self, loss_dict):
        res = None
        if(loss_dict['type']=='CrossEntropyLoss'):
            if loss_dict['class_weight'] is not None:
                res = CrossEntropyLoss(use_sigmoid=loss_dict['use_sigmoid'],loss_weight=loss_dict['loss_weight'],class_weight=loss_dict['class_weight'])
            else :
                 res = CrossEntropyLoss(use_sigmoid=loss_dict['use_sigmoid'],loss_weight=loss_dict['loss_weight'])
        elif (loss_dict['type']=='FocalLoss'):
            if loss_dict['class_weight'] is not None:
                res = FocalLoss(use_sigmoid=loss_dict['use_sigmoid'],loss_weight=loss_dict['loss_weight'],class_weight=loss_dict['class_weight'],loss_name='loss_ce')
            else:
                res = FocalLoss(use_sigmoid=loss_dict['use_sigmoid'],loss_weight=loss_dict['loss_weight'],loss_name='loss_ce')

        else:
            assert False
        assert res!=None
        return res

    def cal_id_loss(self):
        def D_KL_g(mixture,class_i,class_j):
            try:
                radio = (self.diagonal[class_i,mixture]/self.diagonal[class_j,mixture])
                res  = radio.sum(dim=0) - self.embedding_dim + torch.log(radio).sum(dim=0)
                delta_mu = torch.pow(self.means[class_i,mixture]-self.means[class_j,mixture],2)
                res = res + (delta_mu/self.diagonal[class_j,mixture]).sum(dim=0)
                res = res/2
            except:
                print("The diagonal of the covariance matrix cannot be 0, please adjust the transmission again")
                res = 0
            return res
        def D_up(class_i,class_j):
            res = 0
            for m in range(self.num_components):
                res += D_KL_g(m,class_i,class_j)
            return res/self.num_components
        res = 0
        for k in range(self.num_classes):
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i!=j:
                        res += 1/(1+1/2*(D_up(i,j)+D_up(j,i)))
        return res/(self.num_classes*self.num_classes*(self.num_classes-1)) 


    @abstractmethod
    def losses(self, seg_logit, seg_label):
        pass