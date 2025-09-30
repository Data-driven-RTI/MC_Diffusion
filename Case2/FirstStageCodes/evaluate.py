#encoding=utf-8

import torch
import numpy as np
from ignite.metrics import SSIM, PSNR
from ignite.engine import *

if torch.cuda.is_available():
        device = "cuda:0"
else:
        device = "cpu"

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)
ssim_metric = SSIM(data_range=1.6,device=device,gaussian=False)
ssim_metric.attach(default_evaluator,'ssim')
psnr_metric = PSNR(data_range=1.6,device=device)
psnr_metric.attach(default_evaluator,'psnr')

def calculateSSIM_PSNR(prediction,groundtruth):    
    state = default_evaluator.run([[prediction,groundtruth]])
    return state.metrics['ssim'],state.metrics['psnr']

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
     batchSize = O.shape[0]
     ssim_value,psnr_value = calculateSSIM_PSNR(O.unsqueeze(1),F.unsqueeze(1))
     ssim_value *= batchSize
     psnr_value *= batchSize
     uqi_value  = UQI(O,F)
     return ssim_value


