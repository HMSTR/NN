# Параметры сети
import cntk as C
# имя обученной версии
modelName = 'net'
# параметры картинки
channels = 1
width = 33
height = 33
# интерфейсы ИИ
input_size = (channels,width,height)
output_size = 11

# Параметры для обучения
TrainEpochs = 100000
TrainEpochSize = 64
TrainMinibatchSize = 64
# Параметры для проверки
TestEpochSize = 20000
TestMinibatchSize = 16
# Параметры для проверки

def compose(inputs):
    scaled_input = C.ops.element_times(C.ops.constant(0.00390625), inputs)
    with C.layers.default_options(activation=C.ops.relu, pad=False):
        conv1 = C.layers.Convolution2D((15,15), 64, pad=True)(scaled_input)
        pool1 = C.layers.MaxPooling((5,5), (2,2))(conv1)
        conv2 = C.layers.Convolution2D((5,5), 32)(pool1)
        pool2 = C.layers.MaxPooling((3,3), (2,2))(conv2)
        conv3 = C.layers.Convolution2D((3,3), 16)(pool2)
        f4    = C.layers.Dense(512)(conv3)
        drop4 = C.layers.Dropout(0.5)(f4)
        z     = C.layers.Dense(output_size, activation=None)(drop4)
    return z

def composeTrainer(model,outputs):
    lr_per_sample    = [0.001]*100 + [0.0005]*1000 + [0.0001]
    lr_schedule      = C.learning_parameter_schedule_per_sample(lr_per_sample, epoch_size=TrainEpochSize)
    mms = [0]*5 + [0.9990239141819757]
    mm_schedule      = C.learners.momentum_schedule_per_sample(mms, epoch_size=TrainEpochSize)

    # Instantiate the trainer object to drive the model training
    learner = C.learners.momentum_sgd(model.parameters, lr_schedule, mm_schedule)
    training = C.cross_entropy_with_softmax(model, outputs)
    testing = C.classification_error(model, outputs)
    trainer = C.Trainer(model,(training, testing),[learner])
    return trainer



def compose3(inputs):
    with C.layers.default_options(activation=None, pad=False):
        conv1 = C.layers.Convolution2D((25,25), 64, pad=True)(inputs)
        pool1 = C.layers.MaxPooling((4,4), (2,2))(conv1)
#        conv2 = C.layers.Convolution2D((2,2), 48)(pool1)
        f4    = C.layers.Dense(96)(conv2)
#        drop4 = C.layers.Dropout(0.5)(f4)
        model = C.layers.Dense(output_size, activation=None)(result)
    return model