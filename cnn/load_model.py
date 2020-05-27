#coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, transforms
from PIL import Image, ImageOps

# 定義したネットワークのインポート
from net import Net

# デバイスの指定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルの読み込み
model = Net()
model.load_state_dict(torch.load("./mnist_model.pth", map_location=torch.device(device)))
model = model.eval()

# 画像の読み込み
image = Image.open("./data/2.png")
image = ImageOps.invert(image.convert('L')).resize((28,28))
transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, ), (0.5, ))
                       ])
image = transform(image).float()
image = torch.as_tensor(image)
image = image.unsqueeze(0)

# 予測処理
outputs = model(image)
_, predicted = torch.max(outputs.data, 1)
print("predict => " + str(predicted[0].item()))
