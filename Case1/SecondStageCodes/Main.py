#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from S2_arch import DiffIRS2
import torch.optim as optim
from dataloader import RTIDataset
from evaluate import LoadFileForEva,EvaluateAllRatio
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from EvaluatePixelIoU import PixelCount, IoUCount
import numpy as np
from associate import processData
import os
import math
import sys
import random
sys.path.append("../FirstStageCodes")
from save import saveValid
from changeFile import changeFile

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def visualize(testfile,sencondPretrainfile):
    if not os.path.exists("../Visualizations"):
        os.mkdir("../Visualizations")
    SecondStage_pretrain = torch.load(sencondPretrainfile,map_location=device,weights_only=False)
    testDataset = RTIDataset(testfile)
    testloader = DataLoader(dataset=testDataset,batch_size=2,shuffle=False,num_workers=4,pin_memory=True)
    with torch.no_grad():
        SecondStage_pretrain.eval()
        for testidx,(test_data,test_ground) in enumerate(testloader):
            test_data = test_data.to(device)
            test_ground = test_ground.to(device)
            test_img = test_data
            test_img = test_img.unsqueeze(1)
            test_res = SecondStage_pretrain(test_img,None)
            test_res = test_res.squeeze(1)
            if testidx < 120:
                for kk in range(test_res.shape[0]):
                    imgip = test_res[kk].detach().cpu().numpy()
                    imgt = test_ground[kk].detach().cpu().numpy()
                    np.save("../Visualizations/"+str(testidx)+"_"+str(kk)+".npy",imgip)
                    np.save("../Visualizations/"+str(testidx)+"_"+str(kk)+"_gt.npy",imgt)

def EDECalculate(difference):
    return math.sqrt(difference * 4 / math.pi) * 2 / 10

def IoU_RPD_EDE(testfile,pretrainfile):
    model = torch.load(pretrainfile,map_location=device,weights_only=False)
    testDataset = RTIDataset(testfile)
    testloader = DataLoader(dataset=testDataset,batch_size=4,shuffle=False,num_workers=1,pin_memory=True,drop_last=True)
    count = 0
    pixel_difference_sum = 0.0
    pixel_ratio_difference_sum = 0.0
    Iou_sum = 0.0
    str_count = 0
    with torch.no_grad():
        model.eval()
        for testidx,(testdata,testground) in enumerate(testloader):
            print("test idx: ",testidx)
            testdata = testdata.to(device)
            testdata = testdata.unsqueeze(1)
            testground = testground.to(device)
            test_res = model(testdata,None)
            test_res = test_res.squeeze(1)
            batchSize = testdata.shape[0]
            test_res_cur = []
            for ii in range(test_res.shape[0]):
                test_res_cur.append(torch.from_numpy(processData(test_res[ii].detach().cpu().numpy())))
            test_res = torch.stack(test_res_cur,dim=0).to(device)        
            count += batchSize
            pixel_difference,pixel_ratio_difference,difference_list = PixelCount(test_res,testground)
            print("pixel difference: ",pixel_difference, "ratio: ",pixel_ratio_difference)
            pixel_difference_sum += pixel_difference
            pixel_ratio_difference_sum += pixel_ratio_difference
            Iou_sum += IoUCount(test_res,testground)
            print("IoU Batch: ",IoUCount(test_res,testground))

            with open("../SingleTuberError.txt","a+",encoding="utf-8") as fs:
                for dd in difference_list:
                    fs.write(str(str_count))
                    fs.write("\t")
                    fs.write(str(EDECalculate(dd)))
                    fs.write("\n")
                    str_count += 1
                    print(str_count)

        pixel_difference_mean = pixel_difference_sum / count
        pixel_ratio_difference_mean = pixel_ratio_difference_sum / count
        Iou_mean = Iou_sum / count
        EDE = EDECalculate(pixel_difference_mean)

        with open("../EvaluationResult.txt","a+",encoding="utf-8") as fe:
            fe.write("EDE Result:")
            fe.write("\t")
            fe.write(str(EDE))
            fe.write("\t")
            fe.write("RPD Result:")
            fe.write("\t")
            fe.write(str(pixel_ratio_difference_mean))
            fe.write("\t")
            fe.write("IoU Result:")
            fe.write("\t")
            fe.write(str(Iou_mean))
            fe.write("\n")
        
def Imageevaluate(testfile,pretrainfile):
    testDataset = RTIDataset(testfile)
    testloader = DataLoader(dataset=testDataset,batch_size=4,shuffle=False,num_workers=4,pin_memory=True)
    SecondStage = torch.load(pretrainfile,map_location=device,weights_only=False)
    ssim_sum_u = 0.0
    count = 0
    with torch.no_grad():
        SecondStage.eval()
        curtestloss = 0.0
        for testidx,(testdata,testlabel) in enumerate(testloader):
            print("Testidx: ",testidx)
            testdata = testdata.to(device)
            testlabel = testlabel.to(device)
            testimg = testdata.unsqueeze(1)
            test_out = SecondStage(testimg,None)
            test_out = test_out.squeeze(1)
            batchSize = testdata.shape[0]
            for kk in range(batchSize):
                test_res = test_out[kk].to('cpu').numpy()
                test_test = testlabel[kk].to('cpu').numpy()
                ssim_batcht,_,_ = LoadFileForEva(test_test,test_res)
                ssim_sum_u += ssim_batcht
                count += 1
        ssim_avg_u = ssim_sum_u / count
        with open("../EvaluationResult.txt","a+",encoding="utf-8") as fe:
            fe.write("SSIM:")
            fe.write("\t")
            fe.write(str(ssim_avg_u))
            fe.write("\n")

            
if __name__ == "__main__":
    if not os.path.exists("../Test"):
        os.mkdir("../Test")
    saveValid("../datafiles/Test.txt","../RecordsFirst/MultiBranchCNN.pth","../Test/")
    changeFile("../datafiles/Test.txt","../Test/")
    Imageevaluate("../datafiles/Test_FirstStage.txt","../RecordsSecond/Second2_2.pth")
    IoU_RPD_EDE("../datafiles/Test_FirstStage.txt","../RecordsSecond/Second2_2.pth")
    

    
    
    