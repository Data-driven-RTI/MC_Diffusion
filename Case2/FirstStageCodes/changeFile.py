#encoding=utf-8

'''
This file is used to record the reconstructed images from the 
first stage and its corresponding ground truth.
'''

import os
def changeFile(fileName,filedir):
    with open(fileName,"r",encoding="utf-8") as f:
        allLines = f.readlines()
    count = 0
    fw = open(fileName[:-4]+"_FirstStage.txt","w",encoding="utf-8")
    for line in allLines:
        line = line.strip().split("\t")
        fw.write(os.path.join(filedir,str(count)))
        fw.write("\t")
        fw.write(line[1])
        fw.write("\t")
        fw.write("\n")
        count += 1



