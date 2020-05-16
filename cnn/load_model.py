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
#device = torch.device("cpu")
model = Net()
model.load_state_dict(torch.load("./mnist_model.pth", map_location=lambda storage, loc: storage ))
model = Net().to(device)
model = model.eval()

# 予測処理
def predict(X):
    outputs = model(X)
    _, predicted = torch.max(outputs.data, 1)
    print(str(_))
    return(predicted)
    """
    #with torch.no_grad():
        # テストデータを読み込み
        #X = X.to(device)
        outputs = model(X)
        # 予測正確性と予測した種類を数値で返す
        _, predicted = torch.max(outputs.data, 1)
        print(str(_))
        return(predicted)
    """

# 画像の読み込み
image = Image.open("./data/test0001.png")
#image = image.resize((28,28))
#image = image.convert('L').resize((28,28))
image = ImageOps.invert(image.convert('L')).resize((28,28))
transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, ), (0.5, ))
                       ])
image = transform(image).float()
image = torch.as_tensor(image)
image = image.to(device).unsqueeze(0)

# 検証
print("predict => " + str(predict(image)[0].item()))