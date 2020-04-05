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


# загрузчик крартинок
# Используя ImageDeserializer, подгружаются картинки из папки test
def loadSample(pathToMap):
    transforms = [xforms.scale(width=width, height=height, channels=channels, interpolations='linear')]
    _inputs = C.io.StreamDef(field='image',transforms=transforms)
    _outputs = C.io.StreamDef(field='label',shape=output_size)
    return C.io.MinibatchSource(C.io.ImageDeserializer(pathToMap,C.io.StreamDefs(features=_inputs,labels=_outputs)),randomize=True)

inputs = C.input_variable(input_size,dtype=np.float32,name='inputs')
outputs = C.input_variable(output_size,dtype=np.float32,name='outputs')

model = compose(inputs)
trainer = composeTrainer(model,outputs)


def testAccuracy():
    reader = loadSample("data\\test\\map.txt")
    Done = 0
    _i = 0
    input_map = {
        inputs : reader.streams.features,
        outputs : reader.streams.labels
    }

    metric_numer = 0
    metric_denom = 0
    sample_count = 0
    while sample_count < TestEpochSize:
        current_minibatch_size = min(TestMinibatchSize, TestEpochSize - sample_count)
        data = reader.next_minibatch(current_minibatch_size, input_map=input_map)
        metric_numer += trainer.test_minibatch(data) * current_minibatch_size
        metric_denom += current_minibatch_size
        sample_count += current_minibatch_size
        print(str(sample_count*100//TestEpochSize)+"%")
    result = open("logTest.csv","a")
    result.write("Epoch_size:;"+str(TestEpochSize)+';Minibatch_size:;'+str(TestMinibatchSize)+';Error:;'+str(metric_numer/metric_denom)+";\n")
    result.close()


# Если существует модель, то проверяем
if(os.path.isfile(modelName+'.model')):
    print('Previous model found')
    model = C.Function.load(modelName+'.model')
    trainer.restore_from_checkpoint(modelName+'.dnn')
    print("TEST")
    testAccuracy()
