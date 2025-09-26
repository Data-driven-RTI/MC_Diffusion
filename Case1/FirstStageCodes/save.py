#encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloaderf import RTIDataSet
from backbone import RadioNet
import torch.optim as optim 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from evaluate import EvaluateAllRatio
import math
import os
import random
torch.manual_seed(3447)


epochs = 100
lr = 5e-4
ImglossFunction = nn.MSELoss()

device = "cuda:1"

def saveValid(vaildfile,pretrainmodelfile,savedir):
    validDataset = RTIDataSet(vaildfile)
    validloader = DataLoader(validDataset,batch_size=128,num_workers=4,pin_memory=True,shuffle=False)
    pretrainmodel = torch.load(pretrainmodelfile,weights_only=False)
    count = 0
    with torch.no_grad():
        pretrainmodel.eval()
        for valididx,(validdata,_) in enumerate(validloader):
            print(valididx)
            validdata = validdata.to(device)
            valid_img = pretrainmodel(validdata) 
            batchSize = valid_img.shape[0] 
            for ii in range(batchSize):
                valid_img_single = valid_img[ii].to('cpu')
                torch.save(valid_img_single,savedir+str(count))
                count += 1




  