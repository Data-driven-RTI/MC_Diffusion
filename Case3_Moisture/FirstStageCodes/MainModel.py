#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import EncoderModdule
from Imaging import ImagingSimple
from Separator import SeparatorModule


class classfiModule(nn.Module):
    def __init__(self,feature_length):
        super(classfiModule,self).__init__()
        self.Fc1 = nn.Linear(feature_length,128)
        self.Fc2 = nn.Linear(128,2)
        self.relu = nn.LeakyReLU()
    def forward(self,x):
        return self.Fc2(self.relu(self.Fc1(x)))
    

class MainModel(nn.Module):
    def __init__(self,in_channels,feature_length,img_width,img_height) -> None:
        super(MainModel,self).__init__()
        self.Encoder = EncoderModdule(in_channels=in_channels)
        self.Imaging = ImagingSimple(feature_length,img_width,img_height)
        self.Separator = SeparatorModule(feature_length)
        self.Clssifier = classfiModule(feature_length)
        
    def forward(self,X,Y):
        if Y is not None:
            FX = self.Encoder(X)
            FY = self.Encoder(Y)
            X_c,X_e = self.Separator(FX)
            Y_c,Y_e = self.Separator(FY)
            cls_x_e = self.Clssifier(X_e)
            cls_y_e = self.Clssifier(Y_e)
            cls_x_c = self.Clssifier(X_c)
            cls_y_c = self.Clssifier(Y_c)
            cls_c = torch.cat([cls_x_c,cls_y_c],dim=0)
            cls_e = torch.cat([cls_x_e,cls_y_e],dim=0)
            Img_x = self.Imaging(X_c)
            Img_y = self.Imaging(Y_c)
            return X_c,Y_c,Img_x,Img_y,cls_c,cls_e
        else:
            FX = self.Encoder(X)
            X_c,X_e = self.Separator(FX)
            Img_x = self.Imaging(X_c)
            return Img_x 

    


