import numpy as np
from numpy.core.records import array
import tensorflow as tf
import matplotlib.pyplot as plt

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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

import seaborn as sns
import argparse
import datetime
import cv2
from tensorflow.python.keras import callbacks


# Model / data parameters
num_train = 45000
num_val = 5000
num_test = 10000
num_classes = 10
input_shape = (32, 32, 1)

# # Build the model
# model = Sequential()

# # CNN Layer1
# model.add(Conv2D(8, kernel_size=(3,3), strides = 1, padding='same', input_shape = (32, 32, 1), 
#                  kernel_regularizer=regularizers.l2(0.0001)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# # CNN Layer1
# model.add(Conv2D(8, kernel_size=(3,3), strides = 1, padding='same', 
#                  kernel_regularizer=regularizers.l2(0.0001)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = (2, 2), padding='same'))
# # CNN Layer2
# model.add(Conv2D(16, kernel_size=(3,3), strides = 1, padding='same', ))
#                 #  kernel_regularizer=regularizers.l2(0.01)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# # CNN Layer2
# model.add(Conv2D(16, kernel_size=(3,3), strides = 1, padding='same', ))
#                 #  kernel_regularizer=regularizers.l2(0.01)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = (2, 2), padding='same'))
# # CNN Layer3
# model.add(Conv2D(128, kernel_size=(3,3), strides = 1, padding='same', ))
#                 #  kernel_regularizer=regularizers.l2(0.01)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# # CNN Layer3
# model.add(Conv2D(128, kernel_size=(3,3), strides = 1, padding='same', ))
#                 #  kernel_regularizer=regularizers.l2(0.01)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = (2, 2), padding='same'))
# # CNN Layer4
# model.add(Conv2D(256, kernel_size=(3,3), strides = 1, padding='same', ))
#                 #  kernel_regularizer=regularizers.l2(0.01)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# # CNN Layer4
# model.add(Conv2D(256, kernel_size=(3,3), strides = 1, padding='same', ))
#                 #  kernel_regularizer=regularizers.l2(0.01)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = (2, 2), padding='same'))

# # Dense Layer
# model.add(Flatten())
# model.add(Dense(units = 400, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(units = 10, activation='softmax'))

# optimizers = Adam(learning_rate=0.001)
# model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics='accuracy')

## Testing model (double routes)
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import concatenate, Subtract

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
    x1 = Conv2D(8, kernel_size=(3, 3), strides=1, padding='same')(input_1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(8, kernel_size=(3, 3), strides=1, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D(pool_size = (2, 2))(x1)

    x1 = Conv2D(16, kernel_size=(3, 3), strides=1, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(16, kernel_size=(3, 3), strides=1, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D(pool_size = (2, 2))(x1)

    # right route
    x2 = MaxPooling2D(pool_size = (2, 2))(input_1)
    x2 = MaxPooling2D(pool_size = (2, 2))(x2)

    # merge 2 conv outputs
    merge1 = Subtract()([x1, x2])

    # left route
    x1 = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same')(merge1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D(pool_size = (2, 2))(x1)

    x1 = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D(pool_size = (2, 2))(x1)

    # right route
    x2 = MaxPooling2D(pool_size = (2, 2))(merge1)
    x2 = MaxPooling2D(pool_size = (2, 2))(x2)
    x2 = Conv2D(256, kernel_size=(1, 1), strides=1, padding='same')(x2)
    
    # merge 2 conv outputs
    x = Subtract()([x1, x2])

    x = Flatten()(x)
    x = Dense(units = 200, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(10, activation='softmax', name='output')(x)

    model = Model(inputs=[input_1], outputs=[output])
    # model.summary()

    return model

model = multi_route_model()
optimizers = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics='accuracy')

# # ResNet50V2
# res50v2 = ResNet50V2(include_top=False, weights=None, input_tensor=None, input_shape=(32,32,1), classes=10)

# x = res50v2.output
# x = Flatten()(x)

# # 增加 DropOut layer
# x = Dropout(0.5)(x)

# # 增加 Dense layer，以 softmax 產生個類別的機率值
# output_layer = Dense(10, activation='softmax', name='softmax')(x)   
# model = Model(inputs=res50v2.input, outputs=output_layer)
# optimizers = Adam(learning_rate=0.00001)
# model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics='accuracy')


# Command line argument
ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='predict',help='train/predict')
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

    # Save model structure picture
    plot_model(model, show_shapes=True, to_file='run/CIFAR/epochs/model.png')

    # Data input and data augmentation
    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range=0.1,
                                       zoom_range = 0.1, horizontal_flip = True,
                                       rotation_range=20, brightness_range=(0.8, 1.2),
                                       width_shift_range=0.1, height_shift_range=0.1,
                                       zca_whitening=False, fill_mode='nearest')
    val_datagen = ImageDataGenerator(rescale = 1./255)
    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory(
        'CIFAR10/train', 
        target_size = (32, 32), 
        batch_size = batch_size, 
        color_mode="grayscale", 
        class_mode = 'categorical')
    validation_set = val_datagen.flow_from_directory(
        'CIFAR10/val', 
        target_size = (32, 32), 
        batch_size = batch_size, 
        color_mode="grayscale", 
        class_mode = 'categorical')
    test_set = test_datagen.flow_from_directory(
        'CIFAR10/test', 
        target_size = (32, 32), 
        batch_size = batch_size, 
        color_mode="grayscale", 
        class_mode = 'categorical')

    # TensorBoard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    print('\ntensorboard --logdir logs/fit\n')

    # Callback
    EarlyStop = EarlyStopping(monitor='loss', patience=10)
    adapt_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=0.00001)

    # Training
    model_info = model.fit(
        training_set,
        batch_size = batch_size,
        epochs = epoch, 
        verbose = 1, 
        validation_data = validation_set,
        callbacks = [EarlyStop, tensorboard_callback, adapt_lr]
        )

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
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(test_set_X[i],cmap=plt.get_cmap('gray'))
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
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(test_set_X[wrong[i]],cmap=plt.get_cmap('gray'))
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
        plt.xlabel('label: %s, pred: %s' % (getLabel(test_set_y[wrong[i]]), getLabel(res[wrong[i]])), fontsize=14)
        plt.savefig('run/CIFAR/epochs/9Pic_wrong.png')
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix

    print("Building Confusion Matrix...")
    test_set = test_datagen.flow_from_directory(
        'CIFAR10/test', 
        target_size = (32, 32), 
        batch_size = 10000, 
        color_mode="grayscale", 
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

    # # Extract features from specific layer
    # CNN1 = Model(inputs=model.inputs, outputs=model.layers[0].output)
    # CNN2 = Model(inputs=model.inputs, outputs=model.layers[7].output)
    # DNN1 = Model(inputs=model.inputs, outputs=model.layers[29].output)
    # # expand dimensions so that it represents a single 'sample'
    # img = np.expand_dims(test_set_X[0], axis=0)
    # # get feature map for first hidden layer
    # feature1 = CNN1.predict(img)
    # feature2 = CNN2.predict(img)
    # # plot all 28 maps in an 7x4 squares
    # ix = 1
    # plt.figure(figsize=(8,8))
    # for _ in range(2):
    #     for _ in range(4):
    #         # specify subplot and turn of axis
    #         ax = plt.subplot(2, 4, ix)
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         # plot filter channel in grayscale
    #         plt.imshow(feature1[0, :, :, ix-1], cmap='gray')
    #         ix += 1
    # plt.savefig("run/CIFAR/epochs/FeatureCNN1.png")
    # ix = 1
    # plt.figure(figsize=(8,8))
    # for _ in range(4):
    #     for _ in range(4):
    #         # specify subplot and turn of axis
    #         ax = plt.subplot(4, 4, ix)
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         # plot filter channel in grayscale
    #         plt.imshow(feature2[0, :, :, ix-1], cmap='gray')
    #         ix += 1
    # plt.savefig("run/CIFAR/epochs/FeatureCNN2.png")
    # # show the figure
    # plt.show()
