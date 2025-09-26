#encoding=utf-8

import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def SSIMScore(TruthLabel,PredictImg):
    return ssim(PredictImg,TruthLabel,data_range=PredictImg.max()-PredictImg.min())

def PSNRScore(TruthLabel,PredictImg):
    return psnr(TruthLabel,PredictImg)

class ImageEvalue(object):
    def image_mean(self, image):
        mean = np.mean(image)
        return mean
    def image_var(self, image, mean):
        m, n = np.shape(image)
        var = np.sqrt(np.sum((image - mean) ** 2) / (m * n - 1))
        return var
    def images_cov(self, image1, image2, mean1, mean2):
        m, n = np.shape(image1)
        cov = np.sum((image1 - mean1) * (image2 - mean2)) / (m * n - 1)
        return cov
    def UQI(self, O, F):
        meanO = self.image_mean(O)
        meanF = self.image_mean(F)
        varO = self.image_var(O, meanO)
        varF = self.image_var(F, meanF)
        covOF = self.images_cov(O, F, meanO, meanF)
        UQI = 4 * meanO * meanF * covOF / ((meanO ** 2 + meanF ** 2) * (varO ** 2 + varF ** 2))
        return UQI

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
        return torch.sum(UQI).data.item()

def EvaluateAllRatio(O,F):
     uqi_value  = UQI(O,F)
     return uqi_value

evaob = ImageEvalue()
def LoadFileForEva(TrueImage,PredictImage):
    return SSIMScore(TrueImage,PredictImage), PSNRScore(TrueImage,PredictImage), evaob.UQI(TrueImage,PredictImage)




    
   