import cntk as C
import numpy as np
import cv2
import os,sys
import datetime
from random import random, shuffle, randint
import cntk.io.transforms as xforms
from NNconfig import *

print(C.device.all_devices())

C.device.try_set_default_device(C.device.gpu(0))

maxScheme = True

def loadFrames():
    mas = []
    for image in os.listdir('RAW\\'):
        if(image[0]!='.'):
            picture = np.asarray(cv2.imread('RAW\\'+str(image),cv2.IMREAD_GRAYSCALE),order='F')
            edges = np.asarray(cv2.Canny(picture,50,150),order='F')
            mas.append([picture,edges])
    return mas

inputs = C.input_variable(input_size,dtype=np.float32,name='inputs')
outputs = C.input_variable(output_size,dtype=np.float32,name='outputs')

model = compose(inputs)
trainer = composeTrainer(model,outputs)
col = [(255,100,100),(100,255,100),(0,255,0),(0,255,255),(255,0,255),(0,0,255),(255,140,80),(155,155,200),(220,143,220),(179,115,205),(13,235,43),(0,0,0)]

def Work():
    frames = loadFrames()
    for frm in range(0,len(frames)):
        img = frames[frm]
        coords = []
        pool = []
        result = []
        k = 0
        imgFinal = cv2.cvtColor(img[0],cv2.COLOR_GRAY2RGB)
        for shiftY in range(0,287):
            for shiftX in range(0,287):
                cutted = img[0][shiftY:(shiftY+33),shiftX:(shiftX+33)]
                if((np.average(cutted)>30)):
                    # Загружем разрезанные картинки в массив и оптом отправляем на видеокарту считаться
                    edges = img[1][shiftY:(shiftY+33),shiftX:(shiftX+33)]
                    cutted = cv2.add(cutted,edges)
                    coords.append([shiftX,shiftY])
                    pool.append(cutted.reshape(input_size).astype(np.float32))
                    if(len(pool)>7000):
                        print("Done:"+str((frm*100)//len(frames))+"%   "+"lines per frame:"+str((shiftY*100)//287)+"%")
                        result.extend(model.eval(np.asarray(pool)))
                        pool = []

        if(len(pool)>0):
            result.extend(model.eval(np.asarray(pool)))
        if(len(result)>0):
            if(maxScheme):
                bestFrames = np.zeros(output_size).astype(np.int)
                for part in range(0,len(result)):
                    for brainParts in range(0,output_size):
                        bestBrainPartIndex = bestFrames[brainParts]
                        bestBrainPartValue = result[bestBrainPartIndex][brainParts]
                        if(result[part][brainParts]>bestBrainPartValue):
                            bestFrames[brainParts] = part
                if(len(coords)>0):
                    #Для нормальной работы: for x in range(1,output_size-1):
                    for x in range(0,output_size):
                        val = bestFrames[x]
                        XCoord = coords[val][0]
                        YCoord = coords[val][1]
                        A = (XCoord,YCoord)
                        print(A)
                        B = (XCoord+33,YCoord+33)
                        imgFinal = cv2.rectangle(imgFinal,A,B,col[x],1)
            else:
                for inst in range(len(result)):
                    print(result[inst]) 
                    for x in range(1,output_size):
                        
                        if(result[inst][x]>100):

                            frameCoord = coords[inst]
                            color = col[x] 
                            imgFinal = cv2.rectangle(imgFinal,(frameCoord[0],frameCoord[1]),(frameCoord[0]+33,frameCoord[1]+33),color,1)

        cv2.imwrite('OUT\\'+str(frm)+".jpg",imgFinal)
        

if(os.path.isfile(modelName+'.model')):
    print('Previous model found')
    model = C.Function.load(modelName+'.model')
    trainer.restore_from_checkpoint(modelName+'.dnn')
    Work()
else:
    print("There is no model to run")