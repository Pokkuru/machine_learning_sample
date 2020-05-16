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

# モデルの読み込み
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net()
model.load_state_dict(torch.load("./mnist_model.pth"))
model = Net().to(device)
model.eval()

# 予測処理
def predict(X):
    with torch.no_grad():
        # テストデータを読み込み
        X = X.unsqueeze(0)
        X = X.to(device)
        outputs = model(X)
        # 予測正確性と予測した種類を数値で返す
        print(outputs.data)
        _, predicted = torch.max(outputs.data, 1)
        return(predicted)

# 画像の読み込み
image = Image.open("./data/2.png")
# さらにinvertで白黒変換する。画像は文字部分が0(黒)、背景が白(1)で学習元のデータと反対のため。
image = ImageOps.invert(image.convert('L')).resize((28,28))
transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, ), (0.5, ))
                       ])
image = transform(image)


# 検証
print("predict[1.png] => " + str(predict(image)))