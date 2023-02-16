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
from tensorflow.keras.models import Model, load_model
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
from tensorflow.keras.datasets import mnist

import seaborn as sns
import argparse
import cv2
from tensorflow.python.keras import callbacks



(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert Y_train.shape == (60000,)
assert Y_test.shape == (10000,)

# Model / data parameters
num_train = 55000
num_val = 5000
num_test = 10000
num_classes = 10
input_shape = (28, 28, 1)

# Scale images to the [0, 1] range
x_train = X_train.astype("float32") / 255
x_test = X_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Training validation data split
x_train, x_val, y_train, y_val = train_test_split(x_train, Y_train, test_size=num_val)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "training samples")
print(x_val.shape[0], "validation samples")
print(x_test.shape[0], "testing samples")

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(Y_test, num_classes)

# # Build the model
# model = Sequential()

# # CNN Layer1
# model.add(Conv2D(28, kernel_size=(3,3), strides = 1, padding='same', input_shape = (28, 28, 1), ))
#                 #  kernel_regularizer=regularizers.l2(0.01)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# # model.add(MaxPooling2D(pool_size = (2, 2), padding='same'))

# # CNN Layer2
# model.add(Conv2D(56, kernel_size=(2,2), strides = 1, padding='same', ))
#                 #  kernel_regularizer=regularizers.l2(0.01)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = (2, 2), padding='same'))
# # model.add(Dropout(0.2))

# # Dense Layer
# model.add(Flatten())
# model.add(Dense(units = 200, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(units = 10, activation='softmax'))

# optimizers = Adam(learning_rate=0.001)
# model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics='accuracy')

# Build the model
model = Sequential()

# CNN Layer1
model.add(Conv2D(8, kernel_size=(3,3), strides = 2, padding='same', input_shape = (28, 28, 1), 
                 kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# CNN Layer1
model.add(Conv2D(8, kernel_size=(3,3), strides = 1, padding='same', 
                 kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), padding='same'))
# CNN Layer2
model.add(Conv2D(16, kernel_size=(3,3), strides = 1, padding='same', ))
                #  kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# CNN Layer2
model.add(Conv2D(16, kernel_size=(3,3), strides = 1, padding='same', ))
                #  kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), padding='same'))
# CNN Layer3
model.add(Conv2D(32, kernel_size=(3,3), strides = 1, padding='same', ))
                #  kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# CNN Layer3
model.add(Conv2D(32, kernel_size=(3,3), strides = 1, padding='same', ))
                #  kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), padding='same'))
# CNN Layer4
model.add(Conv2D(64, kernel_size=(3,3), strides = 1, padding='same', ))
                #  kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# CNN Layer4
model.add(Conv2D(64, kernel_size=(3,3), strides = 1, padding='same', ))
                #  kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), padding='same'))

# Dense Layer
model.add(Flatten())
model.add(Dense(units = 200, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units = 10, activation='softmax'))

optimizers = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics='accuracy')


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
    plot_model(model, show_shapes=True, to_file='run/MNIST/epochs/model.png')

    # Data input and data augmentation
    train_datagen = ImageDataGenerator(zoom_range = 0.1, rotation_range=5)
    train_datagen.fit(x_train)

    # Callback
    EarlyStop = EarlyStopping(monitor='loss', patience=10)
    adapt_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=0.00001)

    # Training
    model_info = model.fit(
        # x_train, y_train,
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        # batch_size = batch_size,
        epochs = epoch, 
        verbose = 1, 
        shuffle = True,
        validation_data = (x_val, y_val), 
        callbacks = [EarlyStop, adapt_lr]
        )

    # 儲存權重
    model.save_weights('run/MNIST/epochs/last_weights.h5')
    model.save('run/MNIST/epochs/model_keras.h5')
    print('\n====================================================')
    print('Weights are saved to run/train-face/epochs/last_weights.h5')
    print('Best weights are saved to run/train-face/epochs/best_weights.h5')
    print('====================================================\n')

    # Get weights
    layer0CNN, bias = model.layers[0].get_weights()
    layer0CNN = np.array(layer0CNN).reshape(1,-1)

    layer4CNN2, bias = model.layers[7].get_weights()
    layer4CNN2 = np.array(layer4CNN2).reshape(1,-1)

    layer10Dense1, bias = model.layers[29].get_weights()
    layer10Dense1 = np.array(layer10Dense1).reshape(1,-1)

    layer12output, bias = model.layers[31].get_weights()
    layer12output = np.array(layer12output).reshape(1,-1)

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
    plt.savefig('run/MNIST/epochs/History_acc_loss.png')

    plt.show()

    # Weights
    bin_size = 0.005

    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.grid()
    plt.hist(layer0CNN[0,:], bins=np.arange(min(layer0CNN[0,:]), max(layer0CNN[0,:]) + bin_size, bin_size))
    plt.title('Conv1 weights distribution')
    plt.ylabel('Number')
    plt.xlabel('Value')

    plt.subplot(222)
    plt.grid()
    plt.hist(layer4CNN2[0,:], bins=np.arange(min(layer4CNN2[0,:]), max(layer4CNN2[0,:]) + bin_size, bin_size))
    plt.title('Conv2 weights distribution')
    plt.ylabel('Number')
    plt.xlabel('Value')

    plt.subplot(223)
    plt.grid()
    plt.hist(layer10Dense1[0,:], bins=np.arange(min(layer10Dense1[0,:]), max(layer10Dense1[0,:]) + bin_size, bin_size))
    plt.title('Dense1 weights distribution')
    plt.ylabel('Number')
    plt.xlabel('Value')

    plt.subplot(224)
    plt.grid()
    plt.hist(layer12output[0,:], bins=np.arange(min(layer12output[0,:]), max(layer12output[0,:]) + bin_size, bin_size))
    plt.title('Output weights distribution')
    plt.ylabel('Number')
    plt.xlabel('Value')

    plt.tight_layout()
    plt.savefig('run/MNIST/epochs/Weights_distribution.png')

    plt.show()

    # Show some results
    # 結果圖形化
    names = ['0','1','2','3','4','5','6','7','8','9']
    def getLabel(id):
        return ['0','1','2','3','4','5','6','7','8','9'][id]
    res = np.argmax(model.predict(x_test[:9]), axis=-1)

    plt.figure(figsize=(9, 9))
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x_test[i],cmap=plt.get_cmap('gray'))
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
        plt.xlabel('prediction = %s' % getLabel(res[i]), fontsize=14)
        plt.savefig('run/MNIST/epochs/9Pic.png')
    
    # Wrong case
    res = np.argmax(model.predict(x_test), axis=-1)
    # print(res.shape, Y_test.shape)
    wrong = np.where(res != Y_test)
    wrong = np.array(wrong).flatten()
    print("Wrong number:", wrong.shape[0])
    print("Wrong index:", wrong)
    plt.figure(figsize=(9, 9))
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x_test[wrong[i]],cmap=plt.get_cmap('gray'))
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
        plt.xlabel('label: %s, pred: %s' % (Y_test[wrong[i]], getLabel(res[wrong[i]])), fontsize=14)
        plt.savefig('run/MNIST/epochs/9Pic_wrong.png')

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    results = np.argmax(model.predict(x_test), axis=-1)
    cm = confusion_matrix(np.where(y_test == 1)[1], results, normalize='true')
    # plt.matshow(cm)
    # plt.title('Confusion Matrix')
    # plt.clim(0, 1)
    # plt.colorbar()
    # plt.gca().get_xaxis().set_ticks([])
    # plt.gca().get_yaxis().set_ticks([])
    # plt.ylabel('True Label')
    # plt.xlabel('Predicted Label')
    # plt.savefig('run/MNIST/epochs/Confusion_Matrix.png')

    # fig, ax = plt.subplots(figsize=(8,8))
    # ax.matshow(cm, alpha=0.8, vmin=0, vmax=1)
    # for (i, j), z in np.ndenumerate(cm):
    #     ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
    # plt.title('Confusion Matrix')
    # plt.ylabel('True Label')
    # plt.xlabel('Predicted Label')
    # cax = ax.matshow(cm)
    # fig.colorbar(cax)
    # plt.savefig('run/MNIST/epochs/Confusion_Matrix2.png')

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
        horizontalalignment='right'
    )
    plt.savefig("run/MNIST/epochs/Confusion_Matrix.png")
    plt.show()


# elif mode == 'predict':
    # 載入模型
    # model.load_weights(weights_dir)
    # model = load_model(weights_dir)
    # Evaluate with test data
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test accuracy:", score[1])
    print("Test loss:", score[0])

    # Extract features from specific layer
    CNN1 = Model(inputs=model.inputs, outputs=model.layers[0].output)
    CNN2 = Model(inputs=model.inputs, outputs=model.layers[7].output)
    DNN1 = Model(inputs=model.inputs, outputs=model.layers[29].output)
    # expand dimensions so that it represents a single 'sample'
    img = np.expand_dims(x_test[0], axis=0)
    # get feature map for first hidden layer
    feature1 = CNN1.predict(img)
    feature2 = CNN2.predict(img)
    # plot all 28 maps in an 7x4 squares
    ix = 1
    plt.figure(figsize=(8,8))
    for _ in range(2):
        for _ in range(4):
            # specify subplot and turn of axis
            ax = plt.subplot(2, 4, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(feature1[0, :, :, ix-1], cmap='gray')
            ix += 1
    plt.savefig("run/MNIST/epochs/FeatureCNN1.png")
    ix = 1
    plt.figure(figsize=(8,8))
    for _ in range(4):
        for _ in range(4):
            # specify subplot and turn of axis
            ax = plt.subplot(4, 4, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(feature2[0, :, :, ix-1], cmap='gray')
            ix += 1
    plt.savefig("run/MNIST/epochs/FeatureCNN2.png")
    # show the figure
    plt.show()


