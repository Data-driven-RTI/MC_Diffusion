#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from MainModel import MainModel
from dataloaderf import PairTrain,PairTest
import matplotlib.pyplot as plt
import numpy as np



device = "cuda:1"

def saveValid(validfile,pretrainmodelfile,savedir):
    validDataset = PairTest(validfile)
    validloader = DataLoader(validDataset,batch_size=128,num_workers=1,
                             pin_memory=True,shuffle=False)
    pretrainmodel = torch.load(pretrainmodelfile,map_location=device)
    count = 0
    with torch.no_grad():
        pretrainmodel.eval()
        for valididx,(validdata,valground) in enumerate(validloader):
            validdata = validdata.to(device)
            valground = valground.to(device)
            valid_img = pretrainmodel(validdata,None)
            batchSize = validdata.shape[0]
            for ii in range(batchSize):
                valid_img_single = valid_img[ii].to("cpu")
                min_value, max_value = np.min(valid_img_single.numpy()),np.max(valid_img_single.numpy())
                valid_img_single = (valid_img_single-min_value) / (max_value - min_value)
                torch.save(valid_img_single,savedir+str(count)) 
                count += 1
    



   


    
    
    

    

    
