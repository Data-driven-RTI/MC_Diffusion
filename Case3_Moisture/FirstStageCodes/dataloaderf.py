#encoding=utf-8

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np


def dataprocess15(matrix):
    N, M, M = matrix.shape
    diag_mask = torch.eye(M, dtype=torch.bool).unsqueeze(0).expand(N, -1, -1)
    new_matrix = matrix[~diag_mask].view(N, M, -1)
    return new_matrix

class PairTrain(Dataset):
    def __init__(self,fileName) -> None:
        super().__init__()
        self.filenamelist = []
        self.filenamelist2 = []
        self.groundList = []
        with open(fileName,"r",encoding="utf-8") as f:
            flines = f.readlines()
            for line in flines:
                line = line.strip().split("\t")
                self.filenamelist.append(line[0])
                self.filenamelist2.append(line[1])
                self.groundList.append(line[2])
       
    def __getitem__(self, index):
        data = dataprocess15(torch.load(self.filenamelist[index]).float())
        data2 = dataprocess15(torch.load(self.filenamelist2[index]).float())
        ground = torch.from_numpy(np.load(self.groundList[index])).float()
        return torch.abs(data),torch.abs(data2), ground
            
    def __len__(self):
        return len(self.filenamelist)
    

class PairTest(Dataset):
    def __init__(self,fileName) -> None:
        super().__init__()
        self.filenamelist = []
        self.groundList = []
        with open(fileName,"r",encoding="utf-8") as f:
            flines = f.readlines()
            for line in flines:
                line = line.strip().split("\t")
                self.filenamelist.append(line[0])
                self.groundList.append(line[1])
    
    def __getitem__(self, index):
        data = dataprocess15(torch.load(self.filenamelist[index]).float())
        ground = torch.from_numpy(np.load(self.groundList[index])).float()
        return torch.abs(data), ground  
            
    def __len__(self):
        return len(self.filenamelist)
    

