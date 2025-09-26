#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImagingSimple(nn.Module):
    def __init__(self,feature_length,img_width,img_height) -> None:
        super(ImagingSimple,self).__init__()
        self.feature_length = feature_length
        self.img_width = img_width
        self.img_height = img_height
        self.Fc1 = nn.Linear(feature_length,14400) 
        self.relu = nn.LeakyReLU()
    def forward(self,X):
        res = self.relu(self.Fc1(X))
        res = res.reshape(-1,1,120,120)
        res = F.interpolate(res,scale_factor=3,mode="bilinear")
        return res.squeeze(1)

