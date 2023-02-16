import matplotlib.pyplot as plt
import numpy as np
import random
import os
import time

from polars import Float32

import seaborn as sns
import argparse
import datetime
import cv2
from sympy import public

import tensorflow as tf
from tensorflow.keras import initializers
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
from tensorflow.keras.models import load_model

from tensorflow.python.keras.layers.advanced_activations import ReLU
from tensorflow.python.ops.nn_ops import max_pool2d


temp = 0

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

def get_option():
    # Command line argument
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='predict',help='train/predict/show_layers')
    ap.add_argument('--weights', default='weights/model_keras.h5', help='model.h5 path')
    ap.add_argument('--epochs', type=int, default=200)
    opt = ap.parse_args()
    return opt

def XOR_data_create(num):
    input = []
    output = []
    count = 0
    for i in range(num):
        in1 = np.array((random.randint(0, 1)+(random.random()-0.5)*0.01, 
                        random.randint(0, 1)+(random.random()-0.5)*0.01, 
                        random.randint(-1, 1)+(random.random()-0.5)*0.01, 
                        random.randint(-1, 1)+(random.random()-0.5)*0.01))
        input.append(in1)
        if in1[0] == 0 and in1[1] == 0 and in1[2] == 0:
            count += 1
        if (in1[0] >= 0.5 and in1[1] < 0.5) or (in1[0] < 0.5 and in1[1] >= 0.5):
            output.append((1, 0))
            # output.append(1)
        else:
            output.append((0, 1))
            # output.append(0)
    input = np.asarray(input).astype('float32')
    print(input)
    print("Count:", count)
    time.sleep(2)
    return np.array(input[:, 0]), np.array(input[:, 1]), np.array(input[:, 2]), np.array(input[:, 3]), np.array(output)

def build_model():
    input_shape = 1
    input_1 = Input(input_shape, name='input1')
    input_2 = Input(input_shape, name='input2')
    input_3 = Input(input_shape, name='input3')
    input_4 = Input(input_shape, name='input4')
    x1 = Dense(1, use_bias=False, kernel_initializer='zeros', bias_initializer='zeros')(input_1)
    x2 = Dense(1, use_bias=False, kernel_initializer='zeros', bias_initializer='zeros')(input_2)
    x3 = Dense(1, use_bias=False, kernel_initializer='zeros', bias_initializer='zeros')(input_3)
    x4 = Dense(1, use_bias=False, kernel_initializer='zeros', bias_initializer='zeros')(input_4)
    x0 = Concatenate()([x1, x2, x3, x4])
    x = Dense(2, activation='relu', kernel_initializer='he_normal')(x0)
    # x = Dense(2, activation='relu', kernel_initializer='he_normal')(x)
    # x = Dense(2, activation='relu', kernel_initializer='he_normal')(x)
    x = Concatenate()([x, x0])
    x = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
    output = x
    model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=[output])
    return model

def main(opt):
    print("Run main function")

    if opt.mode == 'train':
        solve_cudnn_error()

        model = build_model()
        optimizers = Adam(learning_rate=0.001)
        # optimizers = SGD(learning_rate=0.05, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics='accuracy')

        input1, input2, input3, input4, output = XOR_data_create(4000)
        val_input0, val_input1, val_input2, val_input3, val_out = XOR_data_create(100)
        print(input1)

        model.summary()
        plot_model(model, show_shapes=True, to_file='model.png')

        print(model.predict(x={'input_1':np.array([0]), 'input_2':np.array([0]), 'input_3':np.array([0]), 'input_4':np.array([0])}))
        
        # Callback func
        # EarlyStop = EarlyStopping(monitor='accuracy', patience=10)
        class CustomCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                global temp 
                metrices = self.model.evaluate(self.validation_data)
                if metrices[1] == 1 and temp == 0:    # Acc == 1
                    temp = epoch
                if temp + 10 == epoch and temp != 0:
                    self.model.stop_training = True

        my_callback = CustomCallback()
        model_info = model.fit(
            {'input_1':input1, 'input_2':input2, 'input_3':input3, 'input_4':input4},
            output,
            # validation_data=({'input_1':val_input0, 'input_2':val_input1, 'input_3':val_input2, 'input_4':val_input3}, val_out),
            verbose=1,
            batch_size=1000,
            epochs=2000, 
            # callbacks = [my_callback]
        )

        if not os.path.exists("run/xor/epochs"):
            os.makedirs("run/xor/epochs/")
        # 繪製訓練 & 驗證的準確率值
        plt.figure(figsize=(8, 8))
        # plt.subplot(211)
        plt.grid()
        plt.plot(model_info.history['accuracy'])
        # plt.plot(model_info.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')

        plt.tight_layout()
        plt.savefig('run/xor/epochs/History_acc_loss.png')
        # plt.show()

        # 儲存權重
        model.save_weights('run/xor/epochs/last_weights.h5')
        model.save('run/xor/epochs/model_keras.h5')
        print('\n====================================================')
        print('Weights are saved to run/xor/epochs/last_weights.h5')
        print('Best weights are saved to run/xor/epochs/model_keras.h5')
        print('====================================================\n')

        O1 = model.predict(x={'input_1':np.array([0]), 'input_2':np.array([0]), 'input_3':np.array([0]), 'input_4':np.array([0])})
        O2 = model.predict(x={'input_1':np.array([1]), 'input_2':np.array([0]), 'input_3':np.array([0]), 'input_4':np.array([0])})
        O3 = model.predict(x={'input_1':np.array([0]), 'input_2':np.array([1]), 'input_3':np.array([0]), 'input_4':np.array([0])})
        O4 = model.predict(x={'input_1':np.array([1]), 'input_2':np.array([1]), 'input_3':np.array([0]), 'input_4':np.array([0])})
        print(f"predict(0, 0): {O1[0, 0]:.6f} {O1[0, 1]:.6f}")
        print(f"predict(0, 1): {O2[0, 0]:.6f} {O2[0, 1]:.6f}")
        print(f"predict(1, 0): {O3[0, 0]:.6f} {O3[0, 1]:.6f}")
        print(f"predict(1, 1): {O4[0, 0]:.6f} {O4[0, 1]:.6f}")
        pred = model.predict(x={'input_1':input1, 'input_2':input2, 'input_3':input3, 'input_4':input4})
        pred = np.array(pred >= 0.5)
        wrong = np.where(output != pred)[0]
        print(wrong)
        # for i in wrong:
        #     print("{} {} {} => {} ans:{}".format(input1[i], input2[i], input3[i], pred[i], output[i]))
        # print()


    elif opt.mode == 'predict':
        
        model = load_model('run/xor/epochs/model_keras.h5')

        input = np.array([0, 0], dtype=Float32)
        for i in range(40):
            if i == 10:
                input[0] = 1
            if i == 20:
                input[0] = 0
                input[1] = 1
            if i == 30:
                input[0] = 1
                input[1] = 1
            
            sign = np.array([2*(random.randint(0, 1)-0.5), 2*(random.randint(0, 1)-0.5)]).astype('float32')
            noise = np.array([random.randint(1000, 10000) * sign[0], random.randint(1000, 10000) * sign[1]], dtype=Float32)
            O1 = model.predict(x={'input_1':np.array([input[0]]), 'input_2':np.array([input[1]]), 'input_3':np.array([noise[0]]), 'input_4':np.array([noise[1]])})
            print(f"predict[{i}]({noise[0]}, {noise[1]}): {O1[0, 0]:.6f} {O1[0, 1]:.6f}")

        for layer in model.layers[:]:
            layer_weights = layer.get_weights()
            print(layer_weights)


if __name__ == "__main__":
    main(get_option())