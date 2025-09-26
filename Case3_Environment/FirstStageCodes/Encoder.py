#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderModdule(nn.Module):
    def __init__(self,in_channels) -> None:
        super(EncoderModdule,self).__init__()
        self.in_channels = in_channels
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=128,kernel_size=(3,3),stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(5,5),stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(5,5),stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=(6,5),stride=1),
        )

        # middle convolution
        self.mid_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=256,kernel_size=(9,9),stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(7,7),stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=(2,1),stride=1),
        )
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=512,kernel_size=(16,15),stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=(1,1),stride=1), 
        )
        
    def forward(self,data):
        global_feature_map = self.global_conv(data)
        local_feature_map = self.local_conv(data)
        middle_feature_map = self.mid_conv(data)
        res_feature = global_feature_map + local_feature_map + middle_feature_map 
        res_feature = torch.flatten(res_feature,start_dim=1)
        return res_feature


