import numpy as np
from numpy.core.records import array
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split

from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import LambdaCallback, Callback
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.datasets import cifar10

import seaborn as sns
import argparse
import datetime
import cv2
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.layers.advanced_activations import ReLU
from tensorflow.python.keras.layers.convolutional import Conv, Conv1D
from tensorflow.python.keras.layers.pooling import AveragePooling2D


# Model / data parameters
num_train = 45000
num_val = 5000
num_test = 10000
num_classes = 10
input_shape = (32, 32, 3)

def solve_cudnn_error():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print("\n",len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs\n")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

solve_cudnn_error()

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
# from keras.engine.base_layer import Layer
# from keras import backend as K

class Mish(Layer):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Mish()(X_input)
    '''

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        base_config = super(Mish, self).get_config()
        return {**base_config}

    def compute_output_shape(self, input_shape):
        return input_shape

## Testing model (double routes)
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import concatenate, Subtract, Add, Concatenate
from tensorflow.keras.regularizers import l2

def CNN_Layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu'):
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))
    x = inputs
    x = conv(x)
    x = BatchNormalization()(x)
    # x = Activation(activation)(x)
    x = Mish()(x)
    return x

def Decide_block(inputs,
                 num_filters,
                 kernel_size,
                 strides=1,
                 activation='relu',
                 use_softmax=False):
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='valid',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))
    x = inputs
    x = conv(x)
    x = BatchNormalization()(x)
    x = Mish()(x)
    x = Flatten()(x)
    if use_softmax:
        x = Dropout(0.25)(x)
        x = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)
    return x

def multi_route_model():
    '''
    建構雙路模型
    O ____
    |     | 
    O     |
    |     |
    O     |
    | ____|
    O
    '''
    input_1 = Input(input_shape, name='input1')

    # left route
    x1 = CNN_Layer(input_1, 8)
    x1 = CNN_Layer(x1, 8)
    x1 = CNN_Layer(x1, 16)
    x1 = CNN_Layer(x1, 16)
    x1 = CNN_Layer(x1, 32)
    x1 = CNN_Layer(x1, 32)
    x1 = MaxPooling2D(pool_size = (2, 2))(x1)
    # right route
    x2 = Conv2D(32, kernel_size=(1, 1), strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(input_1)
    x2 = AveragePooling2D(pool_size = (2, 2))(x2)
    # merge 2 conv outputs
    merge1 = Subtract()([x1, x2])
    merge1 = ReLU()(merge1)
    output1 = Decide_block(merge1, 10, kernel_size=16)

    # left route
    x1 = CNN_Layer(merge1, 64)
    x1 = CNN_Layer(x1, 64)
    x1 = CNN_Layer(x1, 128)
    x1 = CNN_Layer(x1, 128)
    x1 = MaxPooling2D(pool_size = (2, 2))(x1)
    # right route
    x2 = AveragePooling2D(pool_size = (2, 2))(merge1)
    x2 = Conv2D(128, kernel_size=(1, 1), strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x2)
    # merge 2 conv outputs
    merge2 = Subtract()([x1, x2])
    merge2 = ReLU()(merge2)
    output2 = Decide_block(merge2, 10, kernel_size=8)

    # left route
    x1 = CNN_Layer(merge2, 256)
    x1 = CNN_Layer(x1, 256)
    x1 = CNN_Layer(x1, 512)
    # right route
    x2 = Conv2D(512, kernel_size=(1, 1), strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(merge2)
    # merge 2 conv outputs
    merge3 = Subtract()([x1, x2])
    merge3 = ReLU()(merge3)
    output3 = Decide_block(merge3, 10, kernel_size=8)

    # left route
    x1 = CNN_Layer(merge3, 512)
    x1 = CNN_Layer(x1, 256)
    x1 = CNN_Layer(x1, 256)
    # right route
    x2 = Conv2D(256, kernel_size=(1, 1), strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(merge3)
    # merge 2 conv outputs
    merge3 = Subtract()([x1, x2])
    merge3 = ReLU()(merge3)
    output6 = Decide_block(merge3, 10, kernel_size=8)

    # left route
    x1 = CNN_Layer(merge3, 128)
    x1 = CNN_Layer(x1, 128)
    x1 = CNN_Layer(x1, 64)
    x1 = CNN_Layer(x1, 64)
    x1 = MaxPooling2D(pool_size = (2, 2))(x1)
    # right route
    x2 = AveragePooling2D(pool_size = (2, 2))(merge3)
    x2 = Conv2D(64, kernel_size=(1, 1), strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x2)
    # merge 2 conv outputs
    merge4 = Subtract()([x1, x2])
    merge4 = ReLU()(merge4)
    output4 = Decide_block(merge4, 10, kernel_size=4)

    # left route
    x1 = CNN_Layer(merge4, 32)
    x1 = CNN_Layer(x1, 32)
    x1 = CNN_Layer(x1, 16)
    x1 = CNN_Layer(x1, 16)
    x1 = CNN_Layer(x1, 8)
    x1 = CNN_Layer(x1, 8)
    x1 = MaxPooling2D(pool_size = (2, 2))(x1)
    # right route
    x2 = AveragePooling2D(pool_size = (2, 2))(merge4)
    x2 = Conv2D(8, kernel_size=(1, 1), strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x2)
    # merge 2 conv outputs
    merge5 = Subtract()([x1, x2])
    merge5 = ReLU()(merge5)
    output5 = Decide_block(merge5, 10, kernel_size=2)

    # Add all decisions
    # x = Add()([output1, output2, output3, output4, output5])
    x = Concatenate()([output1, output2, output3, output4, output5, output6])
    x = Dropout(0.25)(x)
    x = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)
    output = x

    # x = Flatten()(x)
    # # x = Dense(units = 200, activation='relu')(x)
    # x = Dropout(0.25)(x)

    # output = Dense(num_classes, activation='softmax', kernel_initializer='he_normal', name='output')(x)

    model = Model(inputs=[input_1], outputs=[output])
    # model.summary()

    return model

model = multi_route_model()
optimizers = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics='accuracy')

# Command line argument
ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='predict',help='train/predict/show_layers')
ap.add_argument('--weights', default='weights/model_keras.h5', help='model.h5 path')
ap.add_argument('--epochs', type=int, default=30)
opt = ap.parse_args()

# Parameters
batch_size = 64
epoch = opt.epochs
mode = opt.mode
weights_dir = opt.weights

if mode == 'train':
    print("========================================")
    print("Mode =", mode)
    print("Epochs =", epoch)
    print("========================================")

    # Show model info
    model.summary()

    # Save directory
    if not os.path.exists("run/CIFAR/epochs"):
        os.makedirs("run/CIFAR/epochs")

    # Save model structure picture
    plot_model(model, show_shapes=True, to_file='run/CIFAR/epochs/model.png')

    # Data input and data augmentation
    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range=0.1,
                                       zoom_range = 0.1, horizontal_flip = True,
                                       rotation_range=20, brightness_range=(0.8, 1.2),
                                       width_shift_range=0.1, height_shift_range=0.1,
                                       zca_whitening=False, fill_mode='nearest',
                                       zca_epsilon=1e-06, samplewise_center=True)
    val_datagen = ImageDataGenerator(rescale = 1./255, samplewise_center=True)
    test_datagen = ImageDataGenerator(rescale = 1./255, samplewise_center=True)

    training_set = train_datagen.flow_from_directory(
        'CIFAR10/train', 
        target_size = (32, 32), 
        batch_size = batch_size, 
        color_mode="rgb", 
        class_mode = 'categorical')
    validation_set = val_datagen.flow_from_directory(
        'CIFAR10/val', 
        target_size = (32, 32), 
        batch_size = batch_size, 
        color_mode="rgb", 
        class_mode = 'categorical')
    test_set = test_datagen.flow_from_directory(
        'CIFAR10/test', 
        target_size = (32, 32), 
        batch_size = batch_size, 
        color_mode="rgb", 
        class_mode = 'categorical')

    # TensorBoard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    print('\ntensorboard --logdir logs/fit\n')

    # Callback
    save_best = ModelCheckpoint(filepath='run/CIFAR/epochs/best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True)
    EarlyStop = EarlyStopping(monitor='loss', patience=10)
    adapt_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, verbose=1, min_lr=0.000001)

    # Training
    model_info = model.fit(
        training_set,
        batch_size = batch_size,
        epochs = epoch, 
        verbose = 1, 
        validation_data = validation_set,
        callbacks = [EarlyStop, tensorboard_callback, adapt_lr, save_best]
        )
    # K.set_value(model.optimizer.learning_rate, 0.001)
    # print("Learning Rate changed to {} after 10 epochs".format(model.optimizer.learning_rate))

    # model_info = model.fit(
    #     training_set,
    #     batch_size = batch_size,
    #     epochs = epoch-10, 
    #     verbose = 1, 
    #     validation_data = validation_set,
    #     callbacks = [EarlyStop, tensorboard_callback, adapt_lr]
    #     )

    # 儲存權重
    model.save_weights('run/CIFAR/epochs/last_weights.h5')
    model.save('run/CIFAR/epochs/model_keras.h5')
    print('\n====================================================')
    print('Weights are saved to run/train-face/epochs/last_weights.h5')
    print('Best weights are saved to run/train-face/epochs/best_weights.h5')
    print('====================================================\n')

    # # Get weights
    # layer0CNN, bias = model.layers[0].get_weights()
    # layer0CNN = np.array(layer0CNN).reshape(1,-1)

    # layer4CNN2, bias = model.layers[7].get_weights()
    # layer4CNN2 = np.array(layer4CNN2).reshape(1,-1)

    # layer10Dense1, bias = model.layers[29].get_weights()
    # layer10Dense1 = np.array(layer10Dense1).reshape(1,-1)

    # layer12output, bias = model.layers[31].get_weights()
    # layer12output = np.array(layer12output).reshape(1,-1)

    # 繪製訓練 & 驗證的準確率值
    plt.figure(figsize=(8, 8))
    plt.subplot(211)
    plt.grid()
    plt.plot(model_info.history['accuracy'])
    plt.plot(model_info.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    # 繪製訓練 & 驗證的損失值
    # plt.figure()
    plt.subplot(212)
    plt.grid()
    plt.plot(model_info.history['loss'])
    plt.plot(model_info.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    plt.tight_layout()
    plt.savefig('run/CIFAR/epochs/History_acc_loss.png')

    plt.show()

    # # Weights
    # bin_size = 0.005

    # plt.figure(figsize=(8, 8))
    # plt.subplot(221)
    # plt.grid()
    # plt.hist(layer0CNN[0,:], bins=np.arange(min(layer0CNN[0,:]), max(layer0CNN[0,:]) + bin_size, bin_size))
    # plt.title('Conv1 weights distribution')
    # plt.ylabel('Number')
    # plt.xlabel('Value')

    # plt.subplot(222)
    # plt.grid()
    # plt.hist(layer4CNN2[0,:], bins=np.arange(min(layer4CNN2[0,:]), max(layer4CNN2[0,:]) + bin_size, bin_size))
    # plt.title('Conv2 weights distribution')
    # plt.ylabel('Number')
    # plt.xlabel('Value')

    # plt.subplot(223)
    # plt.grid()
    # plt.hist(layer10Dense1[0,:], bins=np.arange(min(layer10Dense1[0,:]), max(layer10Dense1[0,:]) + bin_size, bin_size))
    # plt.title('Dense1 weights distribution')
    # plt.ylabel('Number')
    # plt.xlabel('Value')

    # plt.subplot(224)
    # plt.grid()
    # plt.hist(layer12output[0,:], bins=np.arange(min(layer12output[0,:]), max(layer12output[0,:]) + bin_size, bin_size))
    # plt.title('Output weights distribution')
    # plt.ylabel('Number')
    # plt.xlabel('Value')

    # plt.tight_layout()
    # plt.savefig('run/CIFAR/epochs/Weights_distribution.png')

    # plt.show()

    # Show some results
    # 結果圖形化
    names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    def getLabel(id):
        return ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'][id]
    test_set_X, test_set_y = test_set.next()    # tuple to np.array
    res = np.argmax(model.predict(test_set_X[:9]), axis=-1)
    

    plt.figure(figsize=(9, 9))
    plt.title('9 Case')
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(test_set_X[i])
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
        plt.xlabel('prediction = %s' % getLabel(res[i]), fontsize=14)
        plt.savefig('run/CIFAR/epochs/9Pic.png')
    
    # Wrong case
    res = np.argmax(model.predict(test_set_X), axis=-1)
    test_set_y = np.argmax(test_set_y, axis=-1)
    print("res.shape =", res.shape, "test_set_y.shape =", test_set_y.shape)
    wrong = np.where(res != test_set_y)
    wrong = np.array(wrong).flatten()
    print("Wrong number:", wrong.shape[0], "/", batch_size)
    print("Wrong index:", wrong)
    plt.figure(figsize=(9, 9))
    plt.title('9 Wrong Case')
    if wrong.shape[0] >= 9:
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            plt.imshow(test_set_X[wrong[i]])
            plt.gca().get_xaxis().set_ticks([])
            plt.gca().get_yaxis().set_ticks([])
            plt.xlabel('label: %s, pred: %s' % (getLabel(test_set_y[wrong[i]]), getLabel(res[wrong[i]])), fontsize=14)
            plt.savefig('run/CIFAR/epochs/9Pic_wrong.png')
    else:
        for i in range(0, wrong.shape[0]):
            plt.subplot(330 + 1 + i)
            plt.imshow(test_set_X[wrong[i]])
            plt.gca().get_xaxis().set_ticks([])
            plt.gca().get_yaxis().set_ticks([])
            plt.xlabel('label: %s, pred: %s' % (getLabel(test_set_y[wrong[i]]), getLabel(res[wrong[i]])), fontsize=14)
            plt.savefig('run/CIFAR/epochs/9Pic_wrong.png')
    plt.show()

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix

    print("Building Confusion Matrix...")
    test_set = test_datagen.flow_from_directory(
        'CIFAR10/test', 
        target_size = (32, 32), 
        batch_size = 10000, 
        color_mode="rgb", 
        class_mode = 'categorical')
    test_set_X, test_set_y = test_set.next()    # tuple to np.array
    
    results = np.argmax(model.predict(test_set_X), axis=-1)
    cm = confusion_matrix(np.where(test_set_y == 1)[1], results, normalize='true')
    # plt.matshow(cm)
    # plt.title('Confusion Matrix')
    # plt.clim(0, 1)
    # plt.colorbar()
    # plt.gca().get_xaxis().set_ticks([])
    # plt.gca().get_yaxis().set_ticks([])
    # plt.ylabel('True Label')
    # plt.xlabel('Predicted Label')
    # plt.savefig('run/CIFAR/epochs/Confusion_Matrix.png')

    # fig, ax = plt.subplots(figsize=(8,8))
    # ax.matshow(cm, alpha=0.8, vmin=0, vmax=1)
    # for (i, j), z in np.ndenumerate(cm):
    #     ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
    # plt.title('Confusion Matrix')
    # plt.ylabel('True Label')
    # plt.xlabel('Predicted Label')
    # cax = ax.matshow(cm)
    # fig.colorbar(cax)
    # plt.savefig('run/CIFAR/epochs/Confusion_Matrix2.png')

    # plot heat map
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(
        cm, 
        vmin=0, vmax=1, center=0.5,
        cmap=sns.cubehelix_palette(light=0.4, as_cmap=True),
        annot=True,
        square=True,
        fmt=".3f"
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        horizontalalignment='right',
    )
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.savefig("run/CIFAR/epochs/Confusion_Matrix.png")
    plt.show()

    print("Confusion Matrix Built!")


# elif mode == 'predict':
    # 載入模型
    # model.load_weights(weights_dir)
    # Evaluate with test data
    score = model.evaluate(test_set_X, test_set_y, verbose=0)
    print("Test accuracy:", score[1])
    print("Test loss:", score[0])


elif mode == 'show_layers':
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras import models
    # model = load_model('run/CIFAR/multi_route_test/epochs_imgSubconv4/model_keras.h5')
    model.load_weights('run/CIFAR/epochs/best_model.h5')
    model.summary()

    layer_outputs = [layer.output for layer in model.layers[:]] # Extracts the outputs of (the top 12 layers) all layers
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

    img = np.expand_dims(cv2.imread('run/CIFAR/test.jpg', cv2.IMREAD_GRAYSCALE), axis=-1)
    img = np.expand_dims(img, axis=0)
    activations = activation_model.predict(img) # Returns a list of five Numpy arrays: one array per layer activation

    first_layer_activation = activations[0]
    print("first_layer_activation.shape:", first_layer_activation.shape)

    # plt.figure()
    # plt.matshow(first_layer_activation[0, :, :, 0], cmap='viridis')
    # plt.savefig("run/CIFAR/multi_route_test/epochs_imgSubconv4/feature_map/test.png")
    # # plt.show()

    layer_names = []
    for layer in model.layers[:]:
        layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
        
    images_per_row = 16
    
    if not os.path.exists("run/CIFAR/epochs/feature_map"):
        os.makedirs("run/CIFAR/epochs/feature_map")
    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
        n_features = layer_activation.shape[-1] # Number of features in the feature map
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        if display_grid.shape[1] != 0 and display_grid.shape[0] != 0:
            for col in range(n_cols): # Tiles each filter into a big horizontal grid
                for row in range(images_per_row):
                    channel_image = layer_activation[0,
                                                    :, :,
                                                    col * images_per_row + row]
                    channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size : (col + 1) * size, # Displays the grid
                                row * size : (row + 1) * size] = channel_image
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.matshow(display_grid, cmap='viridis')
            filename = "run/CIFAR/epochs/feature_map/" + str(layer_name) + ".png"
            plt.savefig(filename)
