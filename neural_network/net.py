#coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

# ニューラルネットワークの定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3層のニューラルネットワーク
        self.fc1 = nn.Linear(4, 10) # 入力数4  ニューロン数10
        self.fc2 = nn.Linear(10, 8) # 入力数10 ニューロン数8
        self.fc3 = nn.Linear(8, 3)  # 入力数8  ニューロン数3

    def forward(self, x):
        # 活性化関数はReLUに入力値xを入れる
        # 出力は次の層に渡す
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x