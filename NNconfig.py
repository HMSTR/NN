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
TrainEpochs = 20000
TrainEpochSize = 500
TrainMinibatchSize = 64
# Параметры для проверки
TestEpochSize = 1000
TestMinibatchSize = 16
LRmult = 100
# Параметры для проверки

def compose(inputs):
    with C.layers.default_options(activation=C.ops.relu, pad=False):
        conv1 = C.layers.Convolution2D((25,25), 64, pad=True)(inputs)
        pool1 = C.layers.MaxPooling((4,4), (2,2))(conv1)
        conv2 = C.layers.Convolution2D((2,2), 48)(pool1)
        pool2 = C.layers.MaxPooling((3,3), (2,2))(conv2)
        conv3 = C.layers.Convolution2D((3,3), 64)(pool2)
        f4    = C.layers.Dense(96)(conv3)
        drop4 = C.layers.Dropout(0.5)(f4)
        model = C.layers.Dense(output_size, activation=C.softmax)(drop4)
    return model

def composeTrainer(model,outputs):
    mult = TrainEpochSize//6
    lr_per_sample          = [1]*mult+[0.1]*mult+[0.01]*mult+[0.001]*mult+[0.0001]*mult+[0.00001]*mult+[0.000001]
    lr_schedule            = C.learning_parameter_schedule_per_sample(lr_per_sample, epoch_size=TrainEpochSize)
    l2_reg_weight          = 0.002

    learner = C.learners.adagrad(model.parameters, lr_schedule,l2_regularization_weight = l2_reg_weight)
    training = C.cross_entropy_with_softmax(model, outputs)
    testing = C.classification_error(model, outputs)
    trainer = C.Trainer(model,(training, testing),[learner])
    return trainer