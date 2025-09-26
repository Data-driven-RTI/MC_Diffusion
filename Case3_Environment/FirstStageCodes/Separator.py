#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class SeparatorModule(nn.Module):
    def __init__(self,feature_length) -> None:
        super(SeparatorModule,self).__init__()
        self.ContentFcAtt = nn.Sequential(nn.Linear(feature_length,feature_length//2),nn.Softmax(dim=-1),nn.Linear(feature_length//2,1))
        self.threshold = nn.Linear(feature_length,1)
    def forward(self,X):
        threshold = self.threshold(X)
        mask = (X > threshold).float()
        X = X * mask
        attn_weight = self.ContentFcAtt(X) 
        Content = attn_weight * X
        Environment = X - Content
        return Content, Environment
    

