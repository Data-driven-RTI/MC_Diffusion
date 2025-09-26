#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from S2_arch import DiffIRS2
import torch.optim as optim
from dataloader import RTIDataset
from evaluate import EvaluateAllRatio,LoadFileForEva
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from EvaluatePixelIoU import PixelCount, IoUCount
import numpy as np
from associate import processData
import sys
sys.path.append("../FirstStageCodes")
from save import saveValid
from ChangeFile import changeFile
import os


device = 'cuda:1'


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

def IoU_RPD(testfile,pretrainmodel):
    model = torch.load(pretrainmodel,map_location=device,weights_only=False)
    testDataset = RTIDataset(testfile)
    testloader = DataLoader(dataset=testDataset,batch_size=4,shuffle=False,num_workers=1,pin_memory=True,drop_last=True)
    count = 0
    pixel_ratio_difference_sum = 0.0
    Iou_sum = 0.0
    with torch.no_grad():
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
            _,pixel_ratio_difference = PixelCount(test_res,testground)
            print("RPD: ",pixel_ratio_difference)
            pixel_ratio_difference_sum += pixel_ratio_difference
            Iou_sum += IoUCount(test_res,testground)
            print("IoU Batch: ",IoUCount(test_res,testground))
        pixel_ratio_difference_mean = pixel_ratio_difference_sum / count
        Iou_mean = Iou_sum / count
        with open("../EvaluationResult_Case3_Environment.txt","a+",encoding="utf-8") as fe:
            fe.write("RPD:")
            fe.write("\t")
            fe.write(str(pixel_ratio_difference_mean))
            fe.write("\t")
            fe.write("IoU:")
            fe.write("\t")
            fe.write(str(Iou_mean))
            fe.write("\n")

def ImageEvaluate(testfile,pretrainmodel,strr):
    testDataset = RTIDataset(testfile)
    testloader = DataLoader(dataset=testDataset,batch_size=4,shuffle=False,num_workers=4,pin_memory=True)
    SecondStage = torch.load(pretrainmodel,map_location=device,weights_only=False)
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
        with open("../EvaluationResult_Case3_Environment.txt","a+",encoding="utf-8") as fe:
            fe.write(str(strr))
            fe.write("\n")
            fe.write("SSIM:")
            fe.write("\t")
            fe.write(str(ssim_avg_u))
            fe.write("\n")

            

if __name__ == "__main__":
    
    if not os.path.exists("../TestEhEm_El/"):
        os.mkdir("../TestEhEm_El/")
    saveValid("../datafiles/TestEhEm_El.txt","../RecordsFirst/MultiBranchCNN_EhEm_El.pth","../TestEhEm_El/")
    changeFile("../datafiles/TestEhEm_El.txt","../TestEhEm_El")
    ImageEvaluate("../datafiles/TestEhEm_El_FirstStage.txt","../RecordsSecond/Second2_2_EhEm_El.pth","EhEm_El")
    IoU_RPD("../datafiles/TestEhEm_El_FirstStage.txt","../RecordsSecond/Second2_2_EhEm_El.pth")

    if not os.path.exists("../TestEhEl_Em/"):
        os.mkdir("../TestEhEl_Em/")
    saveValid("../datafiles/TestEhEl_Em.txt","../RecordsFirst/MultiBranchCNN_EhEl_Em.pth","../TestEhEl_Em/")
    changeFile("../datafiles/TestEhEl_Em.txt","../TestEhEl_Em")
    ImageEvaluate("../datafiles/TestEhEl_Em_FirstStage.txt","../RecordsSecond/Second2_2_EhEl_Em.pth","EhEl_Em")
    IoU_RPD("../datafiles/TestEhEl_Em_FirstStage.txt","../RecordsSecond/Second2_2_EhEl_Em.pth")


    if not os.path.exists("../TestEmEl_Eh/"):
        os.mkdir("../TestEmEl_Eh/")
    saveValid("../datafiles/TestEmEl_Eh.txt","../RecordsFirst/MultiBranchCNN_EmEl_Eh.pth","../TestEmEl_Eh/")    
    changeFile("../datafiles/TestEmEl_Eh.txt","../TestEmEl_Eh")
    ImageEvaluate("../datafiles/TestEmEl_Eh_FirstStage.txt","../RecordsSecond/Second2_2_EmEl_Eh.pth","EmEl_Eh")
    IoU_RPD("../datafiles/TestEmEl_Eh_FirstStage.txt","../RecordsSecond/Second2_2_EmEl_Eh.pth")

    
   

