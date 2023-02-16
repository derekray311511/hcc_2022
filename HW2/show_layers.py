import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dla_simple import SimpleDLA
import os, cv2
import matplotlib.pyplot as plt
from torchvision import transforms

weight_dir = "run/epochs/checkpoint/ckpt.pth"
save_dir = "run/epochs/feature_map"
test_img_dir = 'test.jpg'
# weight_dir = "checkpoint/DLA6(data4)/ckpt.pth"
# save_dir = "checkpoint/DLA6(data4)/feature_map"

data_transform = transforms.Compose([transforms.ToPILImage(),
                                    #  transforms.Resize(size=(32, 32)),  # range [0, 255] -> [0.0,1.0] -> [-1.0,1.0]
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)),
                                     ])

device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img0 = cv2.imread(test_img_dir)
img = data_transform(img0)
image = torch.unsqueeze(img,0)
image = image

# 取得每一層的輸出 存到 activation_dic 中
activation_dic = {}
def get_activation(name):
    def hook(model, input, output):
        activation_dic[name] = output.detach()
    return hook

# Load model
model = SimpleDLA()
checkpoint = torch.load(weight_dir)
print("Epcoh =", checkpoint['epoch'])
model.load_state_dict(checkpoint['net']) 

# 保存所有layer的名稱與output值
layer_names = []
activations = []

# print(image.shape)
for name, layer in model.named_modules():
    layer.register_forward_hook(get_activation(name))

# x = torch.randn(1, 3, 64, 64)
# output = model(x)
output = model(image)

for key in activation_dic:
    # print(key)
    # print(np.array(activation[key]))
    layer_names.append(key)
    activations.append(np.array(activation_dic[key]))


images_per_row = 16

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[1] # Number of features in the feature map
    size = layer_activation.shape[2] #The feature map has shape (1, n_features, size, size).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    if display_grid.shape[1] != 0 and display_grid.shape[0] != 0:
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                # print(layer_activation.shape)
                channel_image = layer_activation[0,
                                                col * images_per_row + row,
                                                :, :
                                                ]
                channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                # print(channel_image.shape)
                display_grid[col * size : (col + 1) * size, # Displays the grid
                            row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.matshow(display_grid, cmap='viridis')
        filename = save_dir + "/" + str(layer_name) + ".png"
        plt.savefig(filename)