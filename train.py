import numpy as np
import cv2
import os,sys
import datetime
from random import random, shuffle, randint
import cntk.io.transforms as xforms
from NNconfig import *

print(C.device.all_devices())

C.device.try_set_default_device(C.device.gpu(0))

# загрузчик крартинок
# Используя ImageDeserializer, подгружаются картинки из папки test
def loadSample(pathToMap):
    transforms = [xforms.scale(width=width, height=height, channels=channels, interpolations='linear')]
    _inputs = C.io.StreamDef(field='image',transforms=transforms)
    _outputs = C.io.StreamDef(field='label',shape=output_size)
    return C.io.MinibatchSource(C.io.ImageDeserializer(pathToMap,C.io.StreamDefs(features=_inputs,labels=_outputs)),randomize=True)

# подключаем входные и выходные интерфейсы
inputs = C.input_variable(input_size,dtype=np.float32,name='inputs')
outputs = C.input_variable(output_size,dtype=np.float32,name='outputs')

# загружаем характеристики из NNConfog.py
model = compose(inputs)
trainer = composeTrainer(model,outputs)

#Если найдена предыдущая модель, то будем её дообучать
if(os.path.isfile(modelName+'.model')):
    print('Previous model found')
    model = C.Function.load(modelName+'.model')
    trainer.restore_from_checkpoint(modelName+'.dnn')

def train():
    reader = loadSample("data\\train\\map.txt")
    Done = 0
    _i = 0
    input_map = {
        inputs : reader.streams.features,
        outputs : reader.streams.labels
    }

    for epoch in range(TrainEpochs):
        sample_count = 0
        while sample_count < TrainEpochSize:
            data = reader.next_minibatch(min(TrainMinibatchSize, TrainEpochSize-sample_count), input_map=input_map)
            trainer.train_minibatch(data)
            sample_count += trainer.previous_minibatch_sample_count
        print(str((epoch*100)//TrainEpochs)+"%")    
    model.save(modelName+'.model')
    trainer.save_checkpoint(modelName+'.dnn')
print("TRAIN")
train()