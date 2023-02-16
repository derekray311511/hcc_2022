import os
import numpy as np
from PIL import Image


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = unpickle("data/cifar-10-batches-py/data_batch_1")

labels = data[b'labels']
filenames = data[b'filenames']

classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for i in range(10):
    os.mkdir('./data/CIFAR10/' + classnames[i]) # Build directory
for i in range(len(labels)):
    savefilename = str(filenames[i], encoding='utf-8') # Transform byte format to string format
    saveData = data[i].reshape(3,32,32).astype('uint8')
    R, G, B = Image.fromarray(saveData[0]),\
              Image.fromarray(saveData[1]),\
              Image.fromarray(saveData[2])
    img = Image.merge('RGB', (R, G, B))
    img.save('./data/CIFAR10/' + classnames[labels[i]] + '/' + savefilename)