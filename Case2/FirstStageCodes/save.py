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
from changeFile import changeFile
import math
torch.manual_seed(3447)
import random
import os


ImglossFunction = nn.MSELoss()
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def saveValid(vaildfile,pretrainmodelfile,savedir):
    validDataset = RTIDataSet(vaildfile)
    validloader = DataLoader(validDataset,batch_size=128,num_workers=4,pin_memory=True,shuffle=False)
    pretrainmodel = torch.load(pretrainmodelfile,map_location=device,weights_only=False)
    count = 0
    with torch.no_grad():
        for valididx,(validdata,_) in enumerate(validloader):
            print(valididx)
            validdata = validdata.to(device)
            valid_img = pretrainmodel(validdata) 
            batchSize = valid_img.shape[0] 
            for ii in range(batchSize):
                valid_img_single = valid_img[ii].to('cpu')
                torch.save(valid_img_single,savedir+str(count))
                count += 1
            
def finetune(trainfile,testfile,pretrainfile,strr):
    trainset = RTIDataSet(trainfile)
    testset = RTIDataSet(testfile)
    trainloader = DataLoader(trainset,batch_size=128,pin_memory=True,num_workers=4)
    testloader = DataLoader(testset,batch_size=128,pin_memory=True,num_workers=4)
    model = torch.load(pretrainfile,map_location=device,weights_only=False)
    optimzier = optim.Adam(model.parameters(),lr=5e-4,weight_decay=1e-9)
    bestUQI  = -100000
    for epoch in range(200):
        model.train()
        for trainidx,(traindatax,trainground) in enumerate(trainloader):
            traindatax = traindatax.to(device)
            trainground = trainground.to(device)
            Img_x = model(traindatax)
            img_loss = ImglossFunction(Img_x,trainground)
            allLoss = img_loss
            optimzier.zero_grad()
            allLoss.backward()
            optimzier.step()
            print("Epoch: ",epoch, " Train IDX: ",trainidx, " img loss: ",img_loss.data.item())
        with torch.no_grad():
            model.eval()
            testloss_all = 0.0
            uqi_sum_i = 0.0
            count = 0
            for testidx,(testdatax,testground) in enumerate(testloader):
                testdatax = testdatax.to(device)
                testground = testground.to(device)
                test_img = model(testdatax)
                uqi_batcht = EvaluateAllRatio(testground,test_img)
                uqi_sum_i += uqi_batcht
                testloss_all += F.mse_loss(testground,test_img).data.item()
                count += testdatax.shape[0]
            uqi_avg_i = uqi_sum_i / count
            with open("../RecordsFirst/FineTune_"+strr+".txt","a+",encoding="utf-8") as fe:
                fe.write("EPOCH:")
                fe.write("\t")
                fe.write(str(epoch))
                fe.write("\t")
                fe.write("UQI:")
                fe.write("\t")
                fe.write(str(uqi_avg_i))
                fe.write("\n")
            print("Epoch: ",epoch," UQI: ",uqi_avg_i)
            if uqi_avg_i >= bestUQI:
                bestUQI = uqi_avg_i
                torch.save(model,"../RecordsFirst/FineTune_"+strr+".pth")
                
        
        


            


  