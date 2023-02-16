import matplotlib.pyplot as plt
import numpy as np
import os

import seaborn as sns
import argparse
import datetime
import cv2

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Subtract, Add, Concatenate
from tensorflow.keras.regularizers import l2

from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
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

from tensorflow.python.keras.layers.advanced_activations import ReLU
from tensorflow.python.ops.nn_ops import max_pool2d



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


class BasicBlock(Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2D(planes, kernel_size=3, strides=stride, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(planes, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))
        self.bn2 = BatchNormalization()

        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = Sequential([
                Conv2D(self.expansion*planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
                BatchNormalization()
            ])

    def call(self, x):
        out = ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = ReLU()(out)
        return out


class Root(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Root, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size, strides=1, padding='valid' if kernel_size == 1 else 'same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))
        self.bn = BatchNormalization()

    def call(self, x1, x2):
        x = Concatenate()([x1, x2])
        out = ReLU()(self.bn(self.conv(x)))
        return out


class Tree(Layer):
    def __init__(self, block, in_channels, out_channels, level=1, stride=1):
        super(Tree, self).__init__()
        self.root = Root(2*out_channels, out_channels)
        if level == 1:
            self.left_tree = block(in_channels, out_channels, stride=stride)
            self.right_tree = block(out_channels, out_channels, stride=1)
        else:
            self.left_tree = Tree(block, in_channels,
                                  out_channels, level=level-1, stride=stride)
            self.right_tree = Tree(block, out_channels,
                                   out_channels, level=level-1, stride=1)

    def call(self, x):
        out1 = self.left_tree(x)
        out2 = self.right_tree(out1)
        out = self.root(out1, out2)
        return out


class SimpleDLA(Model):
    def __init__(self, block=BasicBlock, num_classes=10):
        super(SimpleDLA, self).__init__()
        self.base = Sequential([
            Conv2D(16, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            ReLU(),
        ], name='base')

        self.layer1 = Sequential([
            Conv2D(16, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            ReLU()
        ], name='layer1')

        self.layer2 = Sequential([
            Conv2D(32, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            ReLU()
        ], name='layer2')

        self.layer3 = Tree(block,  32,  64, level=1, stride=1)
        self.layer4 = Tree(block,  64, 128, level=2, stride=2)
        self.layer5 = Tree(block, 128, 256, level=2, stride=2)
        self.layer6 = Tree(block, 256, 512, level=1, stride=2)
        self.linear = Dense(num_classes, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))

    def call(self, x):
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = MaxPooling2D(4)(out)
        out = Flatten()(out)
        out = self.linear(out)
        return out

    def model(self):
        input_shape = (32, 32, 3)
        input_1 = Input(input_shape, name = 'input1')
        return Model(inputs=[input_1], outputs=[self.call(input_1)])


x = tf.ones((1, 32, 32, 3))
# x = np.ones((1, 32, 32, 3))

input_shape = (None, 32, 32, 3)
model = SimpleDLA()
model.build(input_shape)
model.model().summary()
# Save model structure picture
plot_model(model.model(), show_shapes=True, to_file='run/epochs/model.png')
y = model(x)
print(y)


optimizers = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics='accuracy')

# Parameters
batch_size = 64
epoch = 10


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
save_best = ModelCheckpoint(filepath='run/epochs/best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True)
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
plt.savefig('run/epochs/History_acc_loss.png')

plt.show()