#encoding=utf-8


import torch
import numpy as np
from ignite.metrics import SSIM, PSNR
from ignite.engine import *



def image_var_torch(image, mean):
        _, m, n = image.shape 
        batch_var = torch.sqrt(torch.sum(torch.sum((image-mean)**2,dim=-1),dim=-1)/(m*n-1))
        return batch_var

def images_cov_torch(image1, image2, mean1, mean2):
        batchsize, m, n = image1.shape
        cov = torch.sum(torch.sum((image1-mean1) * (image2 - mean2),dim=1),dim=-1) / (m*n-1)
        return cov

def UQI(O, F):
        batchsize = O.shape[0]
        meanO = torch.mean(O.reshape(batchsize,-1),dim=-1).reshape(batchsize,1,1)
        meanF = torch.mean(F.reshape(batchsize,-1),dim=-1).reshape(batchsize,1,1)
        varO = image_var_torch(O, meanO)
        varO = varO.reshape(batchsize,1,1)
        varF = image_var_torch(F, meanF)
        varF = varF.reshape(batchsize,1,1)
        covOF = images_cov_torch(O, F, meanO, meanF)
        covOF = covOF.reshape(batchsize,1,1)
        UQI = 4 * meanO * meanF * covOF / ((meanO ** 2 + meanF ** 2) * (varO ** 2 + varF ** 2))
        return torch.sum(UQI)  

def EvaluateAllRatio(O,F):
     uqi_value  = UQI(O,F)
     return uqi_value
