 # -*- coding: utf-8 -*-
from keras.preprocessing.image import save_img
import numpy as np
import pickle

# 解壓縮，返回解壓後的字典
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

# 生成訓練集圖片，如果需要png格式，只需要改圖片字尾名即可。
for j in range(1, 6):
    dataName = "CIFAR10/cifar-10-batches-py/data_batch_" + str(j)  # 讀取當前目錄下的data_batch12345檔案，dataName其實也是data_batch檔案的路徑，本文和指令碼檔案在同一目錄下。
    Xtr = unpickle(dataName)
    print(dataName + " is loading...")

    for i in range(0, 10000):
        img = np.reshape(Xtr['data'][i], (3, 32, 32))  # Xtr['data']為圖片二進位制資料
        img = img.transpose(1, 2, 0)  # 讀取image
        picName = 'CIFAR10/train/' + str(Xtr['labels'][i]) + '_' + str(i + (j - 1)*10000) + '.jpg'  # Xtr['labels']為圖片的標籤，值範圍0-9，本文中，train資料夾需要存在，並與指令碼檔案在同一目錄下。
        save_img(picName, img)
    print(dataName + " loaded.")

print("test_batch is loading...")

# 生成測試集圖片
testXtr = unpickle("CIFAR10/cifar-10-batches-py/test_batch")
for i in range(0, 10000):
    img = np.reshape(testXtr['data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = 'CIFAR10/test/' + str(testXtr['labels'][i]) + '_' + str(i) + '.jpg'
    save_img(picName, img)
print("test_batch loaded.")