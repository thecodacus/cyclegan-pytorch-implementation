from modules.CycleGan import CycleGan
from utils.DataLoader import DataLoader
import numpy as np
import cv2

size=256
imageShape=(size,size,3)
batchSize=1


dl=DataLoader(path="dataset",batchSize=batchSize,imageSize=size*2)

cgan=CycleGan(imageShape[0], imageShape[1], imageShape[2],batchSize=batchSize)
modelSavePath='cgan_saved-163 '+str(size)
cgan.loadModel(modelSavePath)
dataset = dl.getGenerater()
i=0
data=[]
for d in dataset:
    i+=1
    if i==105:
        data=d
        break
datasetX = np.array(data[0])
datasetY = np.array(data[1])
gen,recon=cgan.generate(datasetX.copy())
bGen,bRecon=cgan.bGenerate(datasetY.copy())

print(gen.shape)
cv2.imwrite("testout.jpg", gen[0]) 
cv2.imwrite("testrecon.jpg", recon[0]) 

cv2.imwrite("testBout.jpg", bGen[0]) 
cv2.imwrite("testBrecon.jpg", bRecon[0]) 

cv2.imwrite("testInp.jpg", datasetX[0]) 
cv2.imwrite("testBInp.jpg", datasetY[0]) 