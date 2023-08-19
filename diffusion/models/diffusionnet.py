import math
import random
from typing import List
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
from yolox.models.yolo_pafpn import YOLOPAFPN
from .diffusion_head import DiffusionHead
from yolox.models.network_blocks import BaseConv

class DiffusionNet(nn.Module):
    """
    Implement DiffusionNet
    """

    def __init__(self, backbone=None, head=None, act="silu"):
        super().__init__()
        self.backbone=backbone
        self.head=head
        self.projs=nn.ModuleList()
        in_channels=backbone.in_channels
        for i in range(len(in_channels)):
            self.projs.append(
                BaseConv(
                    in_channels=int(in_channels[i] * head.width),
                    out_channels=int(head.hidden_dim),
                    ksize=1,
                    stride=1,
                    act=act,
                ))

    def forward(self, x, targets=(None,None),random_flip=False,input_size=None):
        # fpn output content features of [dark3, dark4, dark5]
        # x format (pre_imgs,cur_imgs) (B,C,H,W)
        # targets format (pre_targets,cur_targets) (B,N,5) class cx cy w h
        pre_imgs,cur_imgs=x
        pre_targets,cur_targets=targets
        mate_info=(pre_imgs.shape,pre_imgs.device,pre_imgs.dtype)
        bs,_,_,_=mate_info[0]
        if cur_imgs is None:
            x_input=pre_imgs
        else:
            x_input=torch.cat([pre_imgs,cur_imgs],dim=0)

        fpn_outs = self.backbone(x_input)
        flip_mode=False
        if random_flip and torch.randn((1,1))[0]>0.5:
            flip_mode=True
        pre_features,cur_features=[],[]
        
        for proj,x_out in zip(self.projs,fpn_outs):
            l_feat=proj(x_out)
            if cur_imgs is None:
                pre_features.append(l_feat)
                if flip_mode:
                    cur_features.append(torch.flip(l_feat,dims=[3]))
                else:
                    cur_features.append(l_feat.clone())
            else:
                pre_l_feat,cur_l_feat=l_feat.split(bs,dim=0)
                pre_features.append(pre_l_feat)
                cur_features.append(cur_l_feat)

        features=(pre_features,cur_features)

        if self.training:
            assert pre_targets is not None
            if cur_targets is None:
                cur_targets=pre_targets.clone()
                if flip_mode:
                    nlabels=(cur_targets.sum(-1)>0).sum(-1)
                    for idx,nlabel in enumerate(nlabels):
                        cur_targets[idx,:nlabel,1]=input_size[1]-cur_targets[idx,:nlabel,1]
            loss_dict = self.head(
                features,mate_info,targets=torch.cat([pre_targets,cur_targets],dim=0))
            if 'total_loss' not in loss_dict:
                loss_dict['total_loss']=sum(loss_dict.values())
            outputs=loss_dict
            return outputs
        else:  
            outputs = self.head(features,mate_info,targets=pre_targets)

        return outputs


