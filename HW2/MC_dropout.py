# Monte-Carlo Dropout
# web: https://www.cnblogs.com/wuliytTaotao/p/11509634.html
import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

inp = tf.keras.layers.Input(shape=(28, 28))
x = tf.keras.layers.Flatten()(inp)
x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dropout(0.5)(x, training=True)      # dropout 在训练和测试时都将开着
out = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inp, out)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
# 在测试过程，dropout 也是打开的，得到的结果将会有波动，而不是完全一致
print("Len y_test:", len(y_test))
temp = np.zeros((10, len(y_test)))
for i in range(10):
    # with np.printoptions(precision=3, suppress=True):
        # print(model.predict(x_test[:1]))
    temp[i, :] = np.argmax(model.predict(x_test), axis=-1)
count = np.zeros((len(y_test), 10))
for i in range(10):
    for j in range(len(y_test)):
        count[j, int(temp[i, j])] = count[j, int(temp[i, j])] + 1
print(temp)
print(count)
res = np.argmax(count, axis=-1)
print(res)
# Wrong case
# res = np.argmax(model.predict(x_test), axis=-1)
wrong = np.where(res != y_test)
wrong = np.array(wrong).flatten()
print("Wrong number:", wrong.shape[0])
print("Wrong index:", wrong)