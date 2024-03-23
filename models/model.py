        
import time
from functools import partial, reduce

import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool3d
from models.backbones.swin_backbone import swin_3d_small, swin_3d_tiny
from .head import IQAHead, VARHead, VQAHead ,simpleVQAHead
from models.backbones.swin_backbone import SwinTransformer2D as ImageBackbone
from models.backbones.swin_backbone import SwinTransformer3D as VideoBackbone
import torchvision.transforms as T
import numpy as np 
        
class VQA_Network(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()        
        self.config=config
        self.key_names=[]
        self.multi=False
        self.layer=-1
        for key, hypers in config['model']['args'].items():
            
            backbone = swin_3d_small()
            head = VQAHead(**hypers['head'])

            print("Setting backbone:", key + "_backbone")
            setattr(self, key + "_backbone", backbone)
            print("Setting head:", key + "_head")
            setattr(self, key + "_head", head)
    def forward(
        self,
        inputs,
        targets=None,
        inference=True,
        return_pooled_feats=False,
        reduce_scores=False,
        pooled=False,
        clip_return=False,
        **kwargs
    ):
        
     
        scores = []
        feats = {}
        for key in self.key_names:
            feat = getattr(self, key + "_backbone")(
                inputs, multi=self.multi, layer=self.layer, **kwargs
            )
            scores += [getattr(self, key + "_head")(feat)]
            if return_pooled_feats:
                feats[key] = feat
        if reduce_scores:
            if len(scores) > 1:
                scores = reduce(lambda x, y: x + y, scores)
            else:
                scores = scores[0]
        
        if return_pooled_feats:
            return scores, feats
        return scores